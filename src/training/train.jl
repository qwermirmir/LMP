# The training loop over gauge configurations.

# Enable with: LMP.Layer_sequence._TRAIN_ALLOC_DEBUG[] = true
# In this mode gauge JLD2 loads are done synchronously so their allocations
# are attributed to the load step instead of an async task.
const _TRAIN_ALLOC_DEBUG = Ref(false)

function _train_debug_step(label::String, gc0, t0::UInt64; sync_gpu::Bool=false)
    sync_gpu && CUDA.synchronize()
    gc1 = Base.gc_num()
    t1 = time_ns()
    alloc_bytes = max(0, gc1.allocd - gc0.allocd)
    println("[train_debug] ", label,
            " time=", round((t1 - t0) / 1e6; digits=3), "ms",
            " alloc=", round(alloc_bytes / 1024^2; digits=3), "MiB")
    return gc1, t1
end

function train(opt::Opt, layers::Vector{T}, gauge_paths::Vector{String}, U_template, params;
               epochs=100, batch=1, logfile=nothing,
               test_paths::Union{Vector{String},Nothing}=nothing, test_interval=10,
               gauge_cache=nothing, test_gauge_cache=nothing,
               gauge_format::Symbol=:jld2,
               gauge_preprocess! = nothing,
               epoch_offset::Int=0,
               checkpoint_every::Int=0,
               checkpoint_fn=nothing,
               activation_checkpoint_interval::Int=0,
               run_metadata=nothing) where {T<:Layer}

    depth = length(layers)
    N = gauge_format === :raw ? length(gauge_paths) : (gauge_cache === nothing ? length(gauge_paths) : length(gauge_cache))
    loss_history = Float64[]

    if logfile === nothing
        logfile = "train_log_$(Libc.strftime("%Y-%m-%d_%H%M%S", time())).txt"
    end
    log_io = open(logfile, "a")
    _write_training_metadata(log_io, layers, params, gauge_paths, test_paths, gauge_format, opt,
                             epochs, epoch_offset, batch, N, logfile, checkpoint_every,
                             activation_checkpoint_interval, run_metadata)
    println(log_io, "=== Training started: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    println(log_io, "steps=$epochs, batch=$batch, depth=$depth, configs_per_epoch=$N")
    println(log_io, "full_epochs=$((epochs * batch) / N)")
    activation_checkpoint_interval > 0 &&
        println(log_io, "activation_checkpoint_interval=$activation_checkpoint_interval")
    flush(log_io)

    fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)
    test_line = nothing
    if test_paths !== nothing
        test_line, = ax.plot([], [], "r--", linewidth=2, label="test")
        line.set_label("train")
        ax.legend()
    end
    gamma = Clifford()

    # Move the first configuration to GPU.
    _staging = _make_gauge_staging(U_template)
    if gauge_format === :raw
        _raw_cpu_buffers = _make_raw_cpu_buffers(gauge_paths[1], U_template)
        _read_raw_gauge_to_gpu!(U_template, gauge_paths[1], _staging, _raw_cpu_buffers)
    else
        _raw_cpu_buffers = nothing
        first_U = gauge_cache === nothing ? JLD2.load(gauge_paths[1], "U") : gauge_cache[1]
        substitute_U!(U_template, first_U)
        for mu in eachindex(U_template)
            _fast_substitute_U!(U_template[mu], first_U[mu], _staging[mu])
        end
    end
    gauge_preprocess! !== nothing && gauge_preprocess!(U_template)

    CUDA.synchronize()
    x = Initialize_pseudofermion_fields(U_template[1], "Wilson")
    D = Dirac_operator(U_template, x, params)

    # Point PTLayer.U and D at U_template; substitute_U! updates U_template in-place.
    for i in 1:depth
        if layers[i] isa PTLayer
            layers[i].U = U_template
        end
    end
    refresh_model!(layers, D, U_template)

    sync_layers_to_gpu!(layers)
    if activation_checkpoint_interval > 0
        u, delta_tail, transition_j, buf_a, buf_b, delta1_buf =
            _checkpointed_buffer_setup!(layers, D, depth)
    else
        dummy, f = forward_propagation(layers, D, D)
        u = dummy

        # Ping-pong ---------------------------------------------------
        # body_size: PT body length (= pl)
        body_size = length(u[min(4, depth)])

        # Tail: keep separate delta[j] buffers for layers whose size differs from body_size.
        delta_tail = Dict{Int,Any}(depth-1 => deepcopy(u[depth]))
        for j in 3:depth-2
            length(u[j+1]) != body_size && (delta_tail[j] = deepcopy(u[j+1]))
        end

        # transition_j: first tail layer before body ping-pong begins.
        transition_j = minimum(keys(delta_tail))

        # Fill missing tail buffers.
        for j in transition_j:depth-1
            haskey(delta_tail, j) || (delta_tail[j] = deepcopy(u[j+1]))
        end

        # Two ping-pong buffers shaped like u[transition_j].
        buf_a = deepcopy(u[transition_j])
        buf_b = deepcopy(u[transition_j])

        # Layer 1 gradient uses u[2] shape; u[1] is [D], so body ping-pong cannot reuse it.
        delta1_buf = depth >= 2 ? deepcopy(u[2]) : nothing
    end
    # -----------------------------------------------------------------------

    temp = Vector{ComplexF64}(undef, batch)
    d = Vector{Any}(undef, depth)
    test_loss_history = Float64[]
    checkpoint_timer = activation_checkpoint_interval > 0 ? TimerOutput() : nothing
    npad = ndigits(depth)
    fwd_names = activation_checkpoint_interval > 0 ?
        ["fwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth] :
        String[]
    bwd_names = activation_checkpoint_interval > 0 ?
        ["bwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth] :
        String[]
    refresh_error_every_forward =
        lowercase(get(ENV, "COARSE_LOSS_REFRESH_EVERY_FORWARD", "false")) in ("1", "true", "yes", "on")
    refresh_error_every_forward &&
        println(log_io, "coarse_loss_refresh_every_forward=true")

    # D shares U_template and is updated in-place by substitute_U!.
    # Refresh all layer references once more after buffer setup, including
    # nested loss layers such as LossSumLayer(Ritz, CoarseCorrectionLoss).
    refresh_model!(layers, D, U_template)

    function _load_training_gauge!(k::Int)
        if gauge_format === :raw
            _read_raw_gauge_to_gpu!(U_template, gauge_paths[k], _staging, _raw_cpu_buffers)
        elseif gauge_cache !== nothing
            U_src = gauge_cache[k]
            for mu in eachindex(U_template)
                _fast_substitute_U!(U_template[mu], U_src[mu], _staging[mu])
            end
        else
            U_src = JLD2.load(gauge_paths[k], "U")
            for mu in eachindex(U_template)
                _fast_substitute_U!(U_template[mu], U_src[mu], _staging[mu])
            end
        end
        gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
        for i in 1:depth
            layers[i] isa PTLayer && (layers[i].U = U_template)
        end
        refresh_model!(layers, D, U_template)
        return nothing
    end

    for step in 1:epochs
        next_k = mod(step, N) + 1
        current_k = mod1(step, N)
        _train_dbg = _TRAIN_ALLOC_DEBUG[]

        if batch > 1
            activation_checkpoint_interval > 0 &&
                error("ML-style gauge batch currently requires activation_checkpoint_interval=0")

            fill!(d, nothing)
            current_loss = 0.0
            first_k = mod1((epoch_offset + step - 1) * batch + 1, N)
            last_k = first_k
            for b in 1:batch
                sample_index = (epoch_offset + step - 1) * batch + b
                current_k = mod1(sample_index, N)
                last_k = current_k
                _load_training_gauge!(current_k)
                refresh_error_every_forward && refresh_forward_errors!(layers, D)
                f = forward_propagation!(layers, D, D, u)
                temp[b] = f
                current_loss += real(f)
                backward_and_gradient!(layers, u, batch, delta_tail, buf_a, buf_b,
                                       transition_j, d, gamma, delta1_buf)
            end

            advance_step!(opt)
            for i in 1:depth-1
                d[i] === nothing && continue
                update!(layers[i], opt, d[i])
            end

            current_loss /= batch
            push!(real(loss_history), real(current_loss))
            abs_step = epoch_offset + step
            full_epoch = (abs_step * batch) / N
            epoch_text = string(round(full_epoch; digits=4))
            config_text = first_k == last_k ? "$(first_k)/$(N)" : "$(first_k)-$(last_k)/$(N)"
            log_line = "Step: $(abs_step), Epoch: $(epoch_text), ConfigBatch: $(config_text), BatchSize: $(batch), Loss: $(current_loss)$(_loss_metrics_text(layers[end]))"
            println(log_line)
            println(log_io, log_line)
            flush(log_io)

            if test_paths !== nothing && mod(abs_step, test_interval) == 0
                test_loss = 0.0
                n_test = length(test_paths)
                for tc in 1:n_test
                    if gauge_format === :raw
                        _read_raw_gauge_to_gpu!(U_template, test_paths[tc], _staging, _raw_cpu_buffers)
                    else
                        U_src = test_gauge_cache === nothing ? JLD2.load(test_paths[tc], "U") : test_gauge_cache[tc]
                        for mu in eachindex(U_template)
                            _fast_substitute_U!(U_template[mu], U_src[mu], _staging[mu])
                        end
                    end
                    gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
                    refresh_model!(layers, D, U_template)
                    refresh_error_every_forward && refresh_forward_errors!(layers, D)
                    test_loss += real(forward_propagation!(layers, D, D, u))
                end
                test_loss /= n_test
                push!(test_loss_history, real(test_loss))
                test_line_text = "  [Test] Step: $(abs_step), Epoch: $(epoch_text), Test Loss: $(test_loss)$(_loss_metrics_text(layers[end]))"
                println(test_line_text)
                println(log_io, test_line_text)
                flush(log_io)

                test_line !== nothing && test_line.set_xdata(test_interval:test_interval:(test_interval*length(test_loss_history)))
                test_line !== nothing && test_line.set_ydata(test_loss_history)
            end

            if checkpoint_every > 0 && checkpoint_fn !== nothing && abs_step % checkpoint_every == 0
                ckpt = checkpoint_fn(abs_step)
                save_checkpoint(ckpt, layers, opt)
                println("  [Checkpoint] Step $(abs_step) saved: $ckpt")
                println(log_io, "  [Checkpoint] Step $(abs_step) saved: $ckpt")
                flush(log_io)
            end
            continue
        end

        # Load/copy the next gauge configuration.
        if gauge_format === :raw
            _next_U_dbg = nothing
            cpu_prefetch = nothing
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
                _train_debug_step("step $step raw_path_get", _dbg_gc, _dbg_t)
            end
        elseif gauge_cache !== nothing
            _next_U_dbg = gauge_cache[next_k]
            cpu_prefetch = nothing
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
                _train_debug_step("step $step train_cache_get", _dbg_gc, _dbg_t)
            end
        elseif _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            _next_U_dbg = JLD2.load(gauge_paths[next_k], "U")
            _train_debug_step("step $step train_jld2_load", _dbg_gc, _dbg_t)
            cpu_prefetch = nothing
        else
            cpu_prefetch = @async JLD2.load(gauge_paths[next_k], "U")
            _next_U_dbg = nothing
        end

        # Keep PTLayer.U pointing at the shared U_template.
        if _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
        end
        for i in 1:depth
            if layers[i] isa PTLayer
                layers[i].U = U_template
            end
        end
        _train_dbg && _train_debug_step("step $step set_layer_U", _dbg_gc, _dbg_t)


        if _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
        end
        fill!(d, nothing)
        for b in 1:batch
                if activation_checkpoint_interval > 0
                    refresh_error_every_forward && refresh_forward_errors!(layers, D)
                    checkpoints = _checkpointed_forward_loss!(
                    checkpoint_timer, fwd_names, layers, D, u, temp, b, depth,
                    activation_checkpoint_interval)
                _checkpointed_backward_timed!(
                    checkpoint_timer, bwd_names, layers, u, checkpoints,
                    batch, delta_tail, buf_a, buf_b, transition_j, d, gamma,
                    delta1_buf)
                checkpoints = nothing
                _clear_forward_caches!(layers)
                fill!(u, nothing)
                _force_cuda_reclaim!()
                else
                    refresh_error_every_forward && refresh_forward_errors!(layers, D)
                    f = forward_propagation!(layers, D, D, u)
                temp[b] = f
                backward_and_gradient!(layers, u, batch, delta_tail, buf_a, buf_b,
                                       transition_j, d, gamma, delta1_buf)
            end
        end

        _train_dbg && _train_debug_step("step $step fwd_bwd", _dbg_gc, _dbg_t; sync_gpu=true)

        if _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
        end
        advance_step!(opt)
        for i in 1:depth-1
            d[i] === nothing && continue
            update!(layers[i], opt, d[i])
        end
        _train_dbg && _train_debug_step("step $step optimizer", _dbg_gc, _dbg_t; sync_gpu=true)

        if _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
        end
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))

        abs_step = epoch_offset + step
        full_epoch = abs_step / N
        epoch_text = string(round(full_epoch; digits=4))
        log_line = "Step: $(abs_step), Epoch: $(epoch_text), Config: $(current_k)/$(N), Loss: $(current_loss)$(_loss_metrics_text(layers[end], current_loss))"
        println(log_line)
        println(log_io, log_line)
        flush(log_io)
        _train_dbg && _train_debug_step("step $step loss_log", _dbg_gc, _dbg_t)

        if test_paths !== nothing && mod(abs_step, test_interval) == 0
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            end
            test_loss = 0.0
            n_test = length(test_paths)
            test_Us_cpu = gauge_format === :raw ? nothing : Vector{Any}(undef, n_test)
            if gauge_format === :raw
                _train_dbg && _train_debug_step("step $step test_raw_paths_get", _dbg_gc, _dbg_t)
            elseif _train_dbg
                for tc in 1:n_test
                    test_Us_cpu[tc] = test_gauge_cache === nothing ? JLD2.load(test_paths[tc], "U") : test_gauge_cache[tc]
                end
                _train_debug_step("step $step test_jld2_load_all", _dbg_gc, _dbg_t)
            else
                if test_gauge_cache === nothing
                    @sync for tc in 1:n_test
                        Threads.@spawn test_Us_cpu[tc] = JLD2.load(test_paths[tc], "U")
                    end
                else
                    for tc in 1:n_test
                        test_Us_cpu[tc] = test_gauge_cache[tc]
                    end
                end
            end
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            end
            for tc in 1:n_test
                if gauge_format === :raw
                    _read_raw_gauge_to_gpu!(U_template, test_paths[tc], _staging, _raw_cpu_buffers)
                else
                    for mu in eachindex(U_template)
                        _fast_substitute_U!(U_template[mu], test_Us_cpu[tc][mu], _staging[mu])
                    end
                end
                gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
                refresh_model!(layers, D, U_template)
                for b in 1:batch
                    if activation_checkpoint_interval > 0
                        refresh_error_every_forward && refresh_forward_errors!(layers, D)
                        test_temp = Vector{ComplexF64}(undef, 1)
                        checkpoints = _checkpointed_forward_loss!(
                            checkpoint_timer, fwd_names, layers, D, u,
                            test_temp, 1, depth, activation_checkpoint_interval)
                        test_loss += real(test_temp[1])
                        checkpoints = nothing
                        _clear_forward_caches!(layers)
                        fill!(u, nothing)
                        _force_cuda_reclaim!()
                    else
                        refresh_error_every_forward && refresh_forward_errors!(layers, D)
                        test_loss += real(forward_propagation!(layers, D, D, u))
                    end
                end
            end
            _train_dbg && _train_debug_step("step $step test_substitute_forward", _dbg_gc, _dbg_t; sync_gpu=true)
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            end
            test_loss /= n_test
            push!(test_loss_history, real(test_loss))
            test_line_text = "  [Test] Step: $(abs_step), Epoch: $(epoch_text), Test Loss: $(test_loss)$(_loss_metrics_text(layers[end], test_loss))"
            println(test_line_text)
            println(log_io, test_line_text)
            flush(log_io)

            test_line !== nothing && test_line.set_xdata(test_interval:test_interval:(test_interval*length(test_loss_history)))
            test_line !== nothing && test_line.set_ydata(test_loss_history)
            _train_dbg && _train_debug_step("step $step test_log_plotdata", _dbg_gc, _dbg_t)
        end

        if checkpoint_every > 0 && checkpoint_fn !== nothing && abs_step % checkpoint_every == 0
            ckpt = checkpoint_fn(abs_step)
            save_checkpoint(ckpt, layers, opt)
            println("  [Checkpoint] Step $(abs_step) saved: $ckpt")
            println(log_io, "  [Checkpoint] Step $(abs_step) saved: $ckpt")
            flush(log_io)
        end

        # Load the next epoch's U into the shared template.
        if _train_dbg
            _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
        end
        if gauge_format === :raw
            _train_dbg && _train_debug_step("epoch $epoch fetch_next_raw_path", _dbg_gc, _dbg_t)
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            end
            _read_raw_gauge_to_gpu!(U_template, gauge_paths[next_k], _staging, _raw_cpu_buffers)
            gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
            refresh_model!(layers, D, U_template)
            _train_dbg && _train_debug_step("epoch $epoch raw_read_next_U", _dbg_gc, _dbg_t; sync_gpu=true)
        else
            # gauge_cache and the debug path fill _next_U_dbg without a prefetch task.
            _next_U = cpu_prefetch === nothing ? _next_U_dbg : fetch(cpu_prefetch)
            _train_dbg && _train_debug_step("epoch $epoch fetch_next_U", _dbg_gc, _dbg_t)
            if _train_dbg
                _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
            end
            for mu in eachindex(U_template)
                _fast_substitute_U!(U_template[mu], _next_U[mu], _staging[mu])
            end
            gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
            refresh_model!(layers, D, U_template)
            _train_dbg && _train_debug_step("epoch $epoch substitute_next_U", _dbg_gc, _dbg_t; sync_gpu=true)
        end
    end

    if _TRAIN_ALLOC_DEBUG[]
        _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
    end
    sync_layers!(layers)
    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)
    _TRAIN_ALLOC_DEBUG[] && _train_debug_step("finish_sync_log_close", _dbg_gc, _dbg_t; sync_gpu=true)

    if _TRAIN_ALLOC_DEBUG[]
        _dbg_gc = Base.gc_num(); _dbg_t = time_ns()
    end
    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.set_yscale("log")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    _TRAIN_ALLOC_DEBUG[] && _train_debug_step("finish_plot_draw", _dbg_gc, _dbg_t)

end
