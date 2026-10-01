# The training loop over gauge configurations.

"""
    train(opt, layers, gauge_paths, U_template, params; kwargs...) -> loss_history

Trains the model `layers` with the optimizer `opt` on the GPU, cycling through
the gauge configurations in `gauge_paths`. Each step loads a configuration
into `U_template` (the GPU gauge field all layers point at), runs the forward
and backward passes, and takes one Adam step. The loss of every step is
printed and appended to the log file.

`params` holds the Dirac-operator settings passed to `Dirac_operator`, e.g.
`"Dirac_operator" => "Wilson"` and `"κ" => 0.157`.

Keyword arguments:
- `epochs`: number of optimizer steps.
- `batch`: configurations per step. Batches larger than 1 do not average the
  gradient correctly (see `test/test_train_gpu.jl`).
- `gauge_format`: `:jld2` (files with key `"U"`) or `:raw` (files written by
  `convert_gauge_paths_to_raw`, read straight into pinned buffers).
- `gauge_cache`: configurations already in memory, used instead of reading
  `gauge_paths` with `:jld2`.
- `test_paths`, `test_gauge_cache`, `test_interval`: configurations for the
  test loss and how often (in steps) to evaluate it.
- `epoch_offset`: number of steps already done, when resuming.
- `checkpoint_every`, `checkpoint_fn`: save a checkpoint every
  `checkpoint_every` steps to the path `checkpoint_fn(step)`.
- `activation_checkpoint_interval`: keep only every k-th activation and
  recompute the others in the backward pass (0 keeps all).
- `gauge_preprocess!`: called on `U_template` after each load.
- `logfile`, `run_metadata`: log file path and extra lines for its header.

Set the environment variable `COARSE_LOSS_REFRESH_EVERY_FORWARD=true` to
refresh the error vectors of a coarse-correction loss before every forward pass.
"""
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
    test_loss_history = Float64[]
    checkpointing = activation_checkpoint_interval > 0

    # ---- Log file and loss plot ----
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
    checkpointing && println(log_io, "activation_checkpoint_interval=$activation_checkpoint_interval")
    flush(log_io)

    function log_line(text)
        println(text)
        println(log_io, text)
        flush(log_io)
    end

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

    # ---- First configuration and the Dirac operator ----
    staging = _make_gauge_staging(U_template)
    raw_buffers = gauge_format === :raw ? _make_raw_cpu_buffers(gauge_paths[1], U_template) : nothing
    if gauge_format === :raw
        _read_raw_gauge_to_gpu!(U_template, gauge_paths[1], staging, raw_buffers)
    else
        first_U = gauge_cache === nothing ? JLD2.load(gauge_paths[1], "U") : gauge_cache[1]
        substitute_U!(U_template, first_U)
        _copy_gauge_to_template!(U_template, first_U, staging)
    end
    gauge_preprocess! !== nothing && gauge_preprocess!(U_template)

    CUDA.synchronize()
    x = Initialize_pseudofermion_fields(U_template[1], "Wilson")
    D = Dirac_operator(U_template, x, params)

    # All PTLayers and loss layers read U_template, which every load overwrites
    # in place.
    for layer in layers
        layer isa PTLayer && (layer.U = U_template)
    end
    refresh_model!(layers, D, U_template)
    sync_layers_to_gpu!(layers)

    # ---- Buffers for the backward pass ----
    if checkpointing
        u, delta_tail, transition_j, buf_a, buf_b, delta1_buf =
            _checkpointed_buffer_setup!(layers, D, depth)
    else
        u, delta_tail, transition_j, buf_a, buf_b, delta1_buf =
            _backward_buffer_setup!(layers, D, depth)
    end

    gamma = Clifford()
    temp = Vector{ComplexF64}(undef, batch)
    d = Vector{Any}(undef, depth)
    checkpoint_timer = checkpointing ? TimerOutput() : nothing
    npad = ndigits(depth)
    fwd_names = checkpointing ?
        ["fwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth] : String[]
    bwd_names = checkpointing ?
        ["bwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth] : String[]
    refresh_errors =
        lowercase(get(ENV, "COARSE_LOSS_REFRESH_EVERY_FORWARD", "false")) in ("1", "true", "yes", "on")
    refresh_errors && println(log_io, "coarse_loss_refresh_every_forward=true")

    # Refresh once more after the buffer setup, which reaches nested loss
    # layers such as LossSumLayer(Ritz, CoarseCorrectionLoss).
    refresh_model!(layers, D, U_template)

    # ---- Helpers ----

    # Loads training configuration k into U_template.
    function load_training_gauge!(k::Int)
        if gauge_format === :raw
            _read_raw_gauge_to_gpu!(U_template, gauge_paths[k], staging, raw_buffers)
        else
            U_src = gauge_cache !== nothing ? gauge_cache[k] : JLD2.load(gauge_paths[k], "U")
            _copy_gauge_to_template!(U_template, U_src, staging)
        end
        gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
        for layer in layers
            layer isa PTLayer && (layer.U = U_template)
        end
        refresh_model!(layers, D, U_template)
        return nothing
    end

    # One forward and backward pass on the configuration in U_template; adds
    # the gradients into d and returns the loss.
    function forward_backward!(b::Int)
        refresh_errors && refresh_forward_errors!(layers, D)
        if checkpointing
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
            temp[b] = forward_propagation!(layers, D, D, u)
            backward_and_gradient!(layers, u, batch, delta_tail, buf_a, buf_b,
                                   transition_j, d, gamma, delta1_buf)
        end
        return temp[b]
    end

    # Mean loss over the test configurations.
    function test_loss!(abs_step, epoch_text)
        n_test = length(test_paths)
        test_Us = nothing
        if gauge_format !== :raw
            test_Us = Vector{Any}(undef, n_test)
            if test_gauge_cache === nothing
                @sync for tc in 1:n_test
                    Threads.@spawn test_Us[tc] = JLD2.load(test_paths[tc], "U")
                end
            else
                for tc in 1:n_test
                    test_Us[tc] = test_gauge_cache[tc]
                end
            end
        end
        total = 0.0
        for tc in 1:n_test
            if gauge_format === :raw
                _read_raw_gauge_to_gpu!(U_template, test_paths[tc], staging, raw_buffers)
            else
                _copy_gauge_to_template!(U_template, test_Us[tc], staging)
            end
            gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
            refresh_model!(layers, D, U_template)
            refresh_errors && refresh_forward_errors!(layers, D)
            if checkpointing
                test_temp = Vector{ComplexF64}(undef, 1)
                checkpoints = _checkpointed_forward_loss!(
                    checkpoint_timer, fwd_names, layers, D, u,
                    test_temp, 1, depth, activation_checkpoint_interval)
                total += real(test_temp[1])
                checkpoints = nothing
                _clear_forward_caches!(layers)
                fill!(u, nothing)
                _force_cuda_reclaim!()
            else
                total += real(forward_propagation!(layers, D, D, u))
            end
        end
        test_loss = total / n_test
        push!(test_loss_history, real(test_loss))
        metric_loss = batch == 1 ? test_loss : nothing
        log_line("  [Test] Step: $(abs_step), Epoch: $(epoch_text), Test Loss: $(test_loss)$(_loss_metrics_text(layers[end], metric_loss))")
        if test_line !== nothing
            test_line.set_xdata(test_interval:test_interval:(test_interval*length(test_loss_history)))
            test_line.set_ydata(test_loss_history)
        end
        return test_loss
    end

    # ---- Training steps ----
    for step in 1:epochs
        abs_step = epoch_offset + step
        epoch_text = string(round((abs_step * batch) / N; digits=4))
        fill!(d, nothing)

        if batch > 1
            checkpointing &&
                error("ML-style gauge batch currently requires activation_checkpoint_interval=0")
            # Load each configuration of the batch and accumulate its gradient.
            current_loss = 0.0
            first_k = mod1((epoch_offset + step - 1) * batch + 1, N)
            last_k = first_k
            for b in 1:batch
                last_k = mod1((epoch_offset + step - 1) * batch + b, N)
                load_training_gauge!(last_k)
                current_loss += real(forward_backward!(b))
            end
            current_loss /= batch
            config_text = first_k == last_k ? "$(first_k)/$(N)" : "$(first_k)-$(last_k)/$(N)"
            step_text = "Step: $(abs_step), Epoch: $(epoch_text), ConfigBatch: $(config_text), BatchSize: $(batch), Loss: $(current_loss)$(_loss_metrics_text(layers[end]))"
        else
            # U_template already holds this step's configuration. Start
            # reading the next one from disk while the GPU works.
            current_k = mod1(step, N)
            next_k = mod(step, N) + 1
            next_U = nothing
            next_U_task = nothing
            if gauge_format !== :raw
                if gauge_cache !== nothing
                    next_U = gauge_cache[next_k]
                else
                    next_U_task = @async JLD2.load(gauge_paths[next_k], "U")
                end
            end
            for layer in layers
                layer isa PTLayer && (layer.U = U_template)
            end
            forward_backward!(1)
            current_loss = 0.0
            for b in 1:batch
                current_loss += temp[b]
            end
            step_text = "Step: $(abs_step), Epoch: $(epoch_text), Config: $(current_k)/$(N), Loss: $(current_loss)$(_loss_metrics_text(layers[end], current_loss))"
        end

        advance_step!(opt)
        for i in 1:depth-1
            d[i] === nothing && continue
            update!(layers[i], opt, d[i])
        end

        push!(loss_history, real(current_loss))
        log_line(step_text)

        if test_paths !== nothing && mod(abs_step, test_interval) == 0
            test_loss!(abs_step, epoch_text)
        end

        if checkpoint_every > 0 && checkpoint_fn !== nothing && abs_step % checkpoint_every == 0
            ckpt = checkpoint_fn(abs_step)
            save_checkpoint(ckpt, layers, opt)
            log_line("  [Checkpoint] Step $(abs_step) saved: $ckpt")
        end

        # With batch 1, load the configuration of the next step.
        if batch == 1
            if gauge_format === :raw
                _read_raw_gauge_to_gpu!(U_template, gauge_paths[next_k], staging, raw_buffers)
            else
                next_U_task !== nothing && (next_U = fetch(next_U_task))
                _copy_gauge_to_template!(U_template, next_U, staging)
            end
            gauge_preprocess! !== nothing && gauge_preprocess!(U_template)
            refresh_model!(layers, D, U_template)
        end
    end

    sync_layers!(layers)
    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)

    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.set_yscale("log")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return loss_history
end

# Copies a CPU gauge configuration into the GPU template, link by link,
# through the pinned staging buffers.
function _copy_gauge_to_template!(U_template, U_src, staging)
    for mu in eachindex(U_template)
        _fast_substitute_U!(U_template[mu], U_src[mu], staging[mu])
    end
    return U_template
end
