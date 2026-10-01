# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl.

# Set to a layer index j to print per-step @allocated for that layer during backward.
# Example: LMP.Layer_sequence._ALLOC_DEBUG_LAYER[] = 35
const _ALLOC_DEBUG_LAYER = Ref(-1)

# Enable with: LMP.Layer_sequence._GAUGE_COPY_DEBUG[] = true
const _GAUGE_COPY_DEBUG = Ref(false)

# Enable with: LMP.Layer_sequence._OPTIMIZER_DEBUG[] = true
const _OPTIMIZER_DEBUG = Ref(false)

# Enable with: LMP.Layer_sequence._MEM_DEBUG[] = true
# Prints per-phase GPU memory deltas each epoch to isolate 700 MB/epoch growth.
# Uses CUDA.total_memory() - CUDA.available_memory() which matches nvidia-smi.
const _MEM_DEBUG = Ref(false)

function _mem_snapshot()
    GC.gc(true)        # full GC: CuArray finalizer 모두 실행
    CUDA.reclaim()     # pool의 캐싱 메모리를 GPU에 반환
    CUDA.synchronize()
    return CUDA.total_memory() - CUDA.available_memory()
end

# Enable with: LMP.Layer_sequence._GPU_MEMORY_REPORT[] = true
# Prints a live CuArray ownership report from train_timed roots.
const _GPU_MEMORY_REPORT = Ref(false)

_is_gpu_report_leaf(x) =
    x === nothing || x isa Number || x isa Symbol || x isa AbstractString ||
    x isa Type || x isa Function || x isa Module

_gpu_array_bytes(A::CuArray) = Int(length(A)) * sizeof(eltype(A))

function _mark_seen!(seen::IdDict{Any,Bool}, x)
    haskey(seen, x) && return true
    seen[x] = true
    return false
end

function _gpu_collect!(rows::Vector{NamedTuple}, seen_cu::IdDict{Any,Bool},
                       seen_obj::IdDict{Any,Bool}, x, path::String, depth::Int,
                       max_array_items::Int)
    depth < 0 && return nothing
    _is_gpu_report_leaf(x) && return nothing

    if x isa CuArray
        _mark_seen!(seen_cu, x) && return nothing
        push!(rows, (bytes=_gpu_array_bytes(x),
                     path=path,
                     size=string(size(x)),
                     eltype=string(eltype(x))))
        return nothing
    end

    if x isa AbstractDict
        _mark_seen!(seen_obj, x) && return nothing
        for (k, v) in x
            _gpu_collect!(rows, seen_cu, seen_obj, v, path * "[" * sprint(show, k) * "]",
                          depth - 1, max_array_items)
        end
        return nothing
    end

    if x isa Tuple
        for i in eachindex(x)
            _gpu_collect!(rows, seen_cu, seen_obj, x[i], path * "[$i]",
                          depth - 1, max_array_items)
        end
        return nothing
    end

    if x isa AbstractArray
        isbitstype(eltype(x)) && return nothing
        _mark_seen!(seen_obj, x) && return nothing
        n = 0
        for i in eachindex(x)
            n += 1
            n > max_array_items && break
            _gpu_collect!(rows, seen_cu, seen_obj, x[i], path * "[$i]",
                          depth - 1, max_array_items)
        end
        return nothing
    end

    T = typeof(x)
    fieldcount(T) == 0 && return nothing
    _mark_seen!(seen_obj, x) && return nothing
    for (i, name) in enumerate(fieldnames(T))
        isdefined(x, i) || continue
        _gpu_collect!(rows, seen_cu, seen_obj, getfield(x, i),
                      path * "." * String(name), depth - 1, max_array_items)
    end
    return nothing
end

function gpu_memory_report(; roots=[], label::String="", max_items::Int=60,
                           max_depth::Int=14, max_array_items::Int=200)
    CUDA.synchronize()
    rows = NamedTuple[]
    seen_cu = IdDict{Any,Bool}()
    seen_obj = IdDict{Any,Bool}()
    for root in roots
        _gpu_collect!(rows, seen_cu, seen_obj, last(root), String(first(root)),
                      max_depth, max_array_items)
    end
    sort!(rows; by=r -> r.bytes, rev=true)
    total = sum((r.bytes for r in rows); init=0)
    suffix = isempty(label) ? "" : " " * label
    println("[gpu_mem_report]", suffix,
            " unique_cuarrays=", length(rows),
            " total=", round(total / 1024^2; digits=1), "MiB")
    for r in Iterators.take(rows, max_items)
        p = length(r.path) > 180 ? first(r.path, 177) * "..." : r.path
        println("  ", lpad(string(round(r.bytes / 1024^2; digits=1)), 8), " MiB  ",
                p, "  size=", r.size, "  eltype=", r.eltype)
    end
    return rows
end

gpu_memory_report(layers, U_template; kwargs...) =
    gpu_memory_report(; roots=["layers" => layers, "U_template" => U_template], kwargs...)

function gpu_memory_report_bindings(mod::Module=Main; label::String=string(mod) * " bindings",
                                    max_items::Int=60, max_depth::Int=14,
                                    max_array_items::Int=200)
    roots = Pair{String,Any}[]
    for name in names(mod; all=true, imported=false)
        isdefined(mod, name) || continue
        name in (Symbol("#eval"), Symbol("#include")) && continue
        val = getfield(mod, name)
        _is_gpu_report_leaf(val) && continue
        push!(roots, string(mod) * "." * String(name) => val)
    end
    return gpu_memory_report(; roots=roots, label=label, max_items=max_items,
                             max_depth=max_depth, max_array_items=max_array_items)
end

function cuda_memory_summary(label::String="")
    CUDA.synchronize()
    used = CUDA.total_memory() - CUDA.available_memory()
    suffix = isempty(label) ? "" : " " * label
    println("[cuda_mem]", suffix,
            " device_used=", round(Int(used) / 1024^2; digits=1), "MiB",
            " available=", round(Int(CUDA.available_memory()) / 1024^2; digits=1), "MiB",
            " total=", round(Int(CUDA.total_memory()) / 1024^2; digits=1), "MiB")
    return used
end

function cuda_pool_summary(label::String="")
    CUDA.synchronize()
    io = IOBuffer()
    CUDA.pool_status(io)
    suffix = isempty(label) ? "" : " " * label
    for line in split(String(take!(io)), '\n')
        isempty(line) && continue
        println("[cuda_pool]", suffix, " ", line)
    end
    return nothing
end

function _optimizer_debug_step(label::String, gc0, t0::UInt64)
    CUDA.synchronize()
    gc1 = Base.gc_num()
    t1 = time_ns()
    alloc_bytes = max(0, gc1.allocd - gc0.allocd)
    println("[opt_debug] ", label,
            " time=", round((t1 - t0) / 1e6; digits=3), "ms",
            " alloc=", round(alloc_bytes / 1024^2; digits=3), "MiB")
    return gc1, t1
end

function _warm_ritz_wilson_gpu!(D, x)
    x isa WF3x1GPU || return nothing
    tmp1 = _alloc_result_wf(x)
    tmp2 = _alloc_result_wf(x)
    fused_wilson_mul_gpu!(tmp1, x, D.U, _dirac_kappa(D), false)
    fused_wilson_mul_gpu!(tmp2, tmp1, D.U, _dirac_kappa(D), true)
    CUDA.synchronize()
    return nothing
end

# --- GPU -> CPU Hopping_operator ---
function _to_cpu_hop(hop::Hopping_operator)
    Ucpu = _to_cpu_gauge(hop.Gauge)
    if Ucpu === hop.Gauge
        return hop
    end
    return Hopping_operator(Ucpu, hop.P)
end

# --- CPU -> GPU Hopping_operator ---
function _to_gpu_hop(hop::Hopping_operator; blocks=nothing, shared_U=nothing)
    Ugpu = shared_U === nothing ? _to_gpu_gauge(hop.Gauge; blocks=blocks) : shared_U
    if Ugpu === hop.Gauge
        return hop
    end
    return Hopping_operator(Ugpu, hop.P)
end

# --- CPU -> GPU fermion field ---
function _to_gpu_fermion(x::WF3x1CPU; blocks=nothing)
    NC = x.NC
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    if blocks !== nothing
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    xgpu = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
    substitute_fermion!(xgpu, x)
    return xgpu
end

_to_gpu_fermion(x::WF3x1GPU; blocks=nothing) = x

function Lossfunction(f::WF3x1,t::WF3x1)

    loss=f-t
    
    return inner_product(loss,loss)

end

function Lossfunction(f::Vector{WF3x1}, t::Vector{WF3x1})
    @assert length(f) == length(t)
    loss = 0.0
    for i in 1:length(f)
        diff = f[i] - t[i]
        loss += inner_product(diff, diff)
    end
    return loss / length(f)
end

function backward_propagation(layers::Vector{T},  u::Vector{Any}, batch::Int) where T<:Layer
    @assert length(u) == length(layers) 
    @assert all(isassigned(u, j) for j in 1:length(u)) # all u entries must be assigned

    delta = Vector{Any}(undef, length(layers)-1)


    # Note: u[end] is u[L+1], the final output.
    delta[1] = DeltaRule(layers[end], layers[end].activation, u[end], batch)

    if length(layers) > 2
        for i in 1:length(layers)-2
            # Backpropagate from layer L-i using u[L-i].
            delta[1+i] = DeltaRule(
                layers[length(layers) - i],
                layers[length(layers) - i].activation,
                u[length(layers) - i ],
                delta[i]
            )
        end
    end

    @assert length(delta) == length(layers) - 1

    return delta
end

function train(opt::Opt, layers::Vector{T}, input::WF3x1, target::WF3x1 , Dirac_operator ; epochs=100, batch=1,precond=nothing , def=false, logfile=nothing) where {T<:Layer}
    depth = length(layers)
    loss_history = Float64[]

    # Create a logfile when none is provided.
    if logfile === nothing
        logfile = "train_log_$(Libc.strftime("%Y-%m-%d_%H%M%S", time())).txt"
    end
    log_io = open(logfile, "a")
    println(log_io, "=== Training started: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    println(log_io, "epochs=$epochs, batch=$batch, depth=$depth")
    flush(log_io)

    # Live plot for notebook runs.
    fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)

 # display(fig) # Jupyter


    function cosine_annealing_lr(epoch, alpha_max, alpha_min, T_max)
        t = epoch % T_max
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + cos(pi * t / T_max))
    end


    M=precond


    gamma=Clifford()


    D = Dirac_operator

    # Reusable work buffers.
    y = deepcopy(input)
    x = deepcopy(input)

    # Per-epoch forward/backward buffers.
    u = Vector{Any}(undef, depth)
    delta = Vector{Any}(undef, depth-1)
    d = Vector{Any}(undef, depth)
    temp = Vector{ComplexF64}(undef, batch)

    sync_layers_to_gpu!(layers)

    for epoch in 1:epochs

        current_loss = 0.0
        # for i in 1:depth
        #     if layers[i].W === nothing
        #         d[i]=nothing
        #         continue
        #     end
        #     d[i]= similar(layers[i].W)
        # end 

        for b in 1:batch

            # g2vector_generator!(input)
            gauss_distribution_fermion!(input)

            # target=deepcopy(input)/norm(input)

            
            #deflation
            if def== true 
                k=similar(input)
                gmres(input,D,k,maxsteps=1,restart=5)
            end

            if D isa Vector{T}
                y=D(input)
            else 
                mul!(y,D,input)
            end
            # mul!(x,D,y)


            #left
            den=norm(y) 

            #Right
            # den=norm(input)

            # target=input
            target=(input)/den
            if M != nothing
                target=target - M(y/den)
            end
            #left
            input=y/den 
            # input=(y)/den
            # input=x

            # # #Right
            # input=input/den
            
            # input=y-input
            # target=similar(input)


            f = forward_propagation!(layers, input, target, u)
            temp[b] = f

            backward_propagation!(layers, u, batch, delta)

            for i in 1:depth-1
                if b == 1
                    d[i] = gradient(layers[i], layers[i].activation::Any, u[i], delta[i], gamma)
                else
                    # Batch accumulation can be added here.
                    # d[i] = add_gradient(d[i], gradient(...))
                end
            end





      

        end

        # Apply accumulated gradients.
         advance_step!(opt)
         @time for i in 1:depth-1
            # d = gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i + 1],gamma)
            
            # if d === nothing
                # continue # NormLayer
            # end
        

            # Optimizer update.
            update!(layers[i], opt, d[i])

        end

        # Log loss.
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



        println("Epoch: $epoch, Loss: $current_loss")
        println(log_io, "Epoch: $epoch, Loss: $current_loss")
        flush(log_io)



    end
    sync_layers!(layers)
    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)
    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end

function train(opt::Opt, layers::Vector{T} , config ,params ; epochs=100, batch=1, logfile=nothing, test_config=nothing, test_interval=10 ) where {T<:Layer}

    depth = length(layers)

 loss_history = Float64[] #

 # (logfile )
    if logfile === nothing
        logfile = "train_log_$(Libc.strftime("%Y-%m-%d_%H%M%S", time())).txt"
    end
    log_io = open(logfile, "a")
    println(log_io, "=== Training started: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    println(log_io, "epochs=$epochs, batch=$batch, depth=$depth, configs=$(length(config))")
    flush(log_io)

 # test config Dirac operator
    test_Ds = nothing
    test_loss_history = Float64[]
    if test_config !== nothing
        test_Ds = Vector{Any}(undef, length(test_config))
        x_test = Initialize_pseudofermion_fields(test_config[1][1], "Wilson")
        for k in 1:length(test_config)
            test_Ds[k] = Dirac_operator(test_config[k], x_test, params)
        end
    end


        fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
 line, = ax.plot([], [], "b-", linewidth=2) #
    test_line = nothing
    if test_config !== nothing
        test_line, = ax.plot([], [], "r--", linewidth=2, label="test")
        line.set_label("train")
        ax.legend()
    end
    gamma=Clifford()

    x = Initialize_pseudofermion_fields(config[1][1],"Wilson")

 # config Dirac operator ( epoch )
    Ds = Vector{Any}(undef, length(config))
    for k in 1:length(config)
        Ds[k] = Dirac_operator(config[k], x, params)
    end

    D = Ds[1]
    sync_layers_to_gpu!(layers)
    dummy ,f = forward_propagation(layers, D , D)

 delta=deepcopy(dummy[2:end]) # delta
 u = dummy # u ( epoch )

    temp=Vector{ComplexF64}(undef, batch)
    d=Vector{Any}(undef, depth)

    for epoch in 1:epochs
        k=mod(epoch-1,length(config))+1
        D = Ds[k]
        for i in 1:depth
            if layers[i] isa PTLayer
                layers[i].U= config[k]
            end 
        end
        if layers[end] isa PtcRayleighQuotient_Layer
            layers[end].D=D
        elseif  layers[end] isa LossSumLayer
            if layers[end].l1 isa PtcRayleighQuotient_Layer
                layers[end].l1.D=D
            end
            if layers[end].l2 isa PtcRayleighQuotient_Layer
                layers[end].l2.D=D
            end
        end
        if layers[end] isa Ritz_Layer
            layers[end].D=D
        end
        fill!(d, nothing)
        for b in 1:batch
             f = forward_propagation!(layers, D , D, u)
            temp[b]=f
            backward_propagation!(layers, u, batch, delta)



            for i in 1:depth-1
                gi = gradient(layers[i], layers[i].activation, u[i], delta[i], gamma)

                gi === nothing && continue

                if d[i] === nothing
                    d[i] = gi
                else
                    d[i] += gi
                end
            end
        end


 # Gradient grad gradient
         advance_step!(opt)
         for i in 1:depth-1
 # Optimizer
            d[i] === nothing && continue
           update!(layers[i], opt, d[i])
        end

 #
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



 #

 println("Epoch: $(epoch), Loss: $current_loss") #
        println(log_io, "Epoch: $(epoch), Loss: $current_loss")
        flush(log_io)

 # Test loss
        if test_config !== nothing && mod(epoch, test_interval) == 0
            test_loss = 0.0
            for tc in 1:length(test_config)
                D_test = test_Ds[tc]
 # test config layer
                for i in 1:depth
                    if layers[i] isa PTLayer
                        layers[i].U = test_config[tc]
                    end
                end
                if layers[end] isa PtcRayleighQuotient_Layer
                    layers[end].D = D_test
                elseif layers[end] isa LossSumLayer
                    if layers[end].l1 isa PtcRayleighQuotient_Layer
                        layers[end].l1.D = D_test
                    end
                    if layers[end].l2 isa PtcRayleighQuotient_Layer
                        layers[end].l2.D = D_test
                    end
                end
                if layers[end] isa Ritz_Layer
                    layers[end].D = D_test
                end
                for b in 1:batch
                    test_loss += real(forward_propagation!(layers, D_test, D_test, u))
                end
            end
            test_loss /= length(test_config)
            push!(test_loss_history, real(test_loss))
            println("  [Test] Epoch: $(epoch), Test Loss: $test_loss")
            println(log_io, "  [Test] Epoch: $(epoch), Test Loss: $test_loss")
            flush(log_io)

 # test
            test_line.set_xdata(test_interval:test_interval:(test_interval*length(test_loss_history)))
            test_line.set_ydata(test_loss_history)

 # train config
            k_restore = mod(epoch-1, length(config)) + 1
            for i in 1:depth
                if layers[i] isa PTLayer
                    layers[i].U = config[k_restore]
                end
            end
            D_restore = Ds[k_restore]
            if layers[end] isa PtcRayleighQuotient_Layer
                layers[end].D = D_restore
            elseif layers[end] isa LossSumLayer
                if layers[end].l1 isa PtcRayleighQuotient_Layer
                    layers[end].l1.D = D_restore
                end
                if layers[end].l2 isa PtcRayleighQuotient_Layer
                    layers[end].l2.D = D_restore
                end
            end
            if layers[end] isa Ritz_Layer
                layers[end].D = D_restore
            end
        end
    end

    sync_layers!(layers)
    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)

    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
 ax.set_yscale("log") #
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end

# ============================================================
# train_timed: TimerOutputs
#
#    to = train_timed(opt, layers, gauge_paths, U_template, params; epochs=5)
# show(to) #
# reset_timer!(to) #
# ============================================================
function train_timed(opt::Opt, layers::Vector{T}, gauge_paths::Vector{String},
                     U_template, params;
                     epochs::Int=5, batch::Int=1,
                     gauge_format::Symbol=:jld2,
                     checkpoint_interval::Int=0) where {T<:Layer}

    to = TimerOutput()
    depth  = length(layers)
    N      = length(gauge_paths)
    gamma      = Clifford()

    # Initialize staging buffers and load the first configuration.
    _staging = _make_gauge_staging(U_template)
    if gauge_format === :raw
        _raw_cpu_buffers = _make_raw_cpu_buffers(gauge_paths[1], U_template)
        _read_raw_gauge_to_gpu!(U_template, gauge_paths[1], _staging, _raw_cpu_buffers)
    else
        _raw_cpu_buffers = nothing
        first_U = JLD2.load(gauge_paths[1], "U")
        substitute_U!(U_template, first_U)
        for mu in eachindex(U_template)
            _fast_substitute_U!(U_template[mu], first_U[mu], _staging[mu])
        end
    end
    # _sanitize_U!(U_template)
    CUDA.synchronize()

    x = Initialize_pseudofermion_fields(U_template[1], "Wilson")
    D = Dirac_operator(U_template, x, params)

    for i in 1:depth
        layers[i] isa PTLayer && (layers[i].U = U_template)
    end
    if layers[end] isa PtcRayleighQuotient_Layer; layers[end].D = D
    elseif layers[end] isa Ritz_Layer;            layers[end].D = D
    end

    layers[end] isa Ritz_Layer && _warm_ritz_wilson_gpu!(D, x)
    sync_layers_to_gpu!(layers)
    u, _ = forward_propagation(layers, D, D)

    body_size   = length(u[min(4, depth)])
    delta_tail  = Dict{Int,Any}(depth-1 => deepcopy(u[depth]))
    for j in 3:depth-2
        length(u[j+1]) != body_size && (delta_tail[j] = deepcopy(u[j+1]))
    end
    transition_j = minimum(keys(delta_tail))
    for j in transition_j:depth-1
        haskey(delta_tail, j) || (delta_tail[j] = deepcopy(u[j+1]))
    end
    buf_a      = deepcopy(u[transition_j])
    buf_b      = deepcopy(u[transition_j])
    delta1_buf = depth >= 2 ? deepcopy(u[2]) : nothing
    temp       = Vector{ComplexF64}(undef, batch)
    d          = Vector{Any}(undef, depth)

    # Prewarm backward/gradient caches before timing so DeltaRule scratch buffers,
    # inverse path tables, Wadj buffers, and GPU gradient buffers are allocated once.
    fill!(d, nothing)
    forward_propagation!(layers, D, D, u)
    backward_and_gradient!(layers, u, batch, delta_tail, buf_a, buf_b,
                           transition_j, d, gamma, delta1_buf)
    fill!(d, nothing)
    prealloc_optimizer_state!(layers, opt)
    CUDA.synchronize()
    if checkpoint_interval > 0
        _clear_forward_caches!(layers)
        fill!(u, nothing)
        _force_cuda_reclaim!()
    end

    if _GPU_MEMORY_REPORT[]
        cuda_memory_summary("train_timed pre-loop before root report")
        cuda_pool_summary("train_timed pre-loop before root report")
        gpu_memory_report(; roots=[
            "layers" => layers,
            "U_template" => U_template,
            "D" => D,
            "u" => u,
            "delta_tail" => delta_tail,
            "buf_a" => buf_a,
            "buf_b" => buf_b,
            "delta1_buf" => delta1_buf,
            "d" => d,
        ], label="train_timed pre-loop", max_items=120, max_array_items=500)
        gpu_memory_report_bindings(Main; label="Main bindings pre-loop",
                                   max_items=80, max_array_items=500)
        CUDA.reclaim()
        cuda_memory_summary("train_timed pre-loop after CUDA.reclaim")
        cuda_pool_summary("train_timed pre-loop after CUDA.reclaim")
    end

        # Zero gradients and run forward/backward for the batch.
    npad = ndigits(depth)
    fwd_names = ["fwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth]
    bwd_names = ["bwd_" * lpad(i, npad, '0') * "_$(nameof(typeof(layers[i])))" for i in 1:depth]

    println("train_timed: epochs=$epochs, batch=$batch, depth=$depth")
    checkpoint_interval > 0 && println("  checkpoint_interval=$checkpoint_interval")

        # Forward/backward batch timing.
    for epoch in 1:epochs
        next_k = mod(epoch, N) + 1
        cpu_prefetch = gauge_format === :raw ? nothing : @async JLD2.load(gauge_paths[next_k], "U")

        for i in 1:depth
            layers[i] isa PTLayer && (layers[i].U = U_template)
        end

        fill!(d, nothing)
        _mem_epoch_start = _MEM_DEBUG[] ? _mem_snapshot() : UInt(0)
        NVTX.@mark "epoch_$epoch"

        for b in 1:batch
            checkpoints = nothing
            # --- forward ---
            NVTX.@range "fwd" @timeit to "forward" begin
                if checkpoint_interval > 0
                    checkpoints = _checkpointed_forward_loss!(to, fwd_names, layers, D, u, temp, b, depth, checkpoint_interval)
                else
                    u[1] = [D]
                for i in 1:(depth-1)
                    NVTX.@range fwd_names[i] @timeit to fwd_names[i] u[i+1] = layers[i](u[i])
                end
                # Drain accumulated GPU finalizations before the final layer's timing window,
                # so that async-kernel bookkeeping from layers 1..depth-1 is not misattributed
                # to fwd_depth (Ritz_Layer). Only pays the cost when the last layer needs
                # GPU synchronization anyway (i.e., GPU path).
                if layers[depth] isa Ritz_Layer && !isempty(u[depth]) && u[depth][1] isa WF3x1GPU
                    CUDA.synchronize()
                    GC.gc(false)
                end
                    NVTX.@range fwd_names[depth] @timeit to fwd_names[depth] temp[b] = layers[depth](u[depth])
                end
            end
            CUDA.synchronize()

            # --- backward: backward_and_gradient! ---
            NVTX.@range "bwd" @timeit to "backward" begin
                if checkpoint_interval > 0
                    _checkpointed_backward_timed!(to, bwd_names, layers, u, checkpoints,
                                                  batch, delta_tail, buf_a, buf_b,
                                                  transition_j, d, gamma, delta1_buf)
                else
                    L = depth

                # Loss layer.
                NVTX.@range bwd_names[L] @timeit to bwd_names[L] DeltaRule!(layers[L], layers[L].activation, u[L], delta_tail[L-1], batch)

                # Tail: j = L-1 down to transition_j.
                for j in (L-1):-1:transition_j
                    cur = delta_tail[j]
                    NVTX.@range bwd_names[j] @timeit to bwd_names[j] begin
                        if _ALLOC_DEBUG_LAYER[] == j
                            _b0 = Base.gc_num(); gi = gradient(layers[j], layers[j].activation, u[j], cur, gamma); _b1 = Base.gc_num()
                            a1 = max(0, _b1.allocd - _b0.allocd)
                            _b0 = Base.gc_num(); gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi); _b1 = Base.gc_num()
                            a2 = max(0, _b1.allocd - _b0.allocd)
                            nxt = j > transition_j ? delta_tail[j-1] : buf_a
                            _b0 = Base.gc_num(); DeltaRule!(layers[j], layers[j].activation, u[j], cur, nxt); _b1 = Base.gc_num()
                            a3 = max(0, _b1.allocd - _b0.allocd)
                            println("[alloc_diag] bwd_$(lpad(j,2,'0'))_$(nameof(typeof(layers[j])))  grad=$a1 B  accum=$a2 B  delta=$a3 B")
                        else
                            gi = gradient(layers[j], layers[j].activation, u[j], cur, gamma)
                            gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)
                            if j > transition_j
                                DeltaRule!(layers[j], layers[j].activation, u[j], cur, delta_tail[j-1])
                            else
                                DeltaRule!(layers[j], layers[j].activation, u[j], cur, buf_a)
                            end
                        end
                    end
                end

                # Body ping-pong: j = transition_j-1 down to 2.
                cur_buf, nxt_buf = buf_a, buf_b
                for j in (transition_j - 1):-1:2
                    NVTX.@range bwd_names[j] @timeit to bwd_names[j] begin
                        if _ALLOC_DEBUG_LAYER[] == j
                            _b0 = Base.gc_num(); gi = gradient(layers[j], layers[j].activation, u[j], cur_buf, gamma); _b1 = Base.gc_num()
                            a1 = max(0, _b1.allocd - _b0.allocd)
                            _b0 = Base.gc_num(); gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi); _b1 = Base.gc_num()
                            a2 = max(0, _b1.allocd - _b0.allocd)
                            if j > 2
                                _b0 = Base.gc_num(); DeltaRule!(layers[j], layers[j].activation, u[j], cur_buf, nxt_buf); _b1 = Base.gc_num()
                                a3 = max(0, _b1.allocd - _b0.allocd)
                                cur_buf, nxt_buf = nxt_buf, cur_buf
                            else
                                a3 = 0
                            end
                            println("[alloc_diag] bwd_$(lpad(j,2,'0'))_$(nameof(typeof(layers[j])))  grad=$a1 B  accum=$a2 B  delta=$a3 B")
                        else
                            gi = gradient(layers[j], layers[j].activation, u[j], cur_buf, gamma)
                            gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)
                            if j > 2
                                DeltaRule!(layers[j], layers[j].activation, u[j], cur_buf, nxt_buf)
                                cur_buf, nxt_buf = nxt_buf, cur_buf
                            end
                        end
                    end
                end

                # Layer 2 DeltaRule! plus layer 1 gradient.
                if delta1_buf !== nothing && L >= 2
                    NVTX.@range bwd_names[2] @timeit to bwd_names[2] DeltaRule!(layers[2], layers[2].activation, u[2], cur_buf, delta1_buf)
                    NVTX.@range bwd_names[1] @timeit to bwd_names[1] begin
                        gi1 = gradient(layers[1], layers[1].activation, u[1], delta1_buf, gamma)
                        gi1 !== nothing && (d[1] = d[1] === nothing ? gi1 : d[1] + gi1)
                    end
                end
                end
            end
            CUDA.synchronize()
            if checkpoint_interval > 0
                checkpoints = nothing
                _clear_forward_caches!(layers)
                fill!(u, nothing)
                _force_cuda_reclaim!()
            end
        end

        _mem_after_bwd = _MEM_DEBUG[] ? _mem_snapshot() : UInt(0)
        # --- optimizer ---
        NVTX.@range "opt" @timeit to "optimizer" begin
            advance_step!(opt)
            for i in 1:depth-1
                d[i] === nothing && continue
                if _OPTIMIZER_DEBUG[]
                    _opt_gc = Base.gc_num()
                    _opt_t = time_ns()
                    update!(layers[i], opt, d[i])
                    _optimizer_debug_step("update[$i] $(nameof(typeof(layers[i])))", _opt_gc, _opt_t)
                else
                    update!(layers[i], opt, d[i])
                end
            end
            CUDA.synchronize()
        end

        # --- gauge copy (JLD2 fetch + CPU -> GPU copy) ---
        _mem_after_opt = _MEM_DEBUG[] ? _mem_snapshot() : UInt(0)
        NVTX.@range "gauge" @timeit to "gauge_copy" begin
            if gauge_format === :raw
                @timeit to "raw_read_copy" begin
                    _read_raw_gauge_to_gpu!(U_template, gauge_paths[next_k], _staging, _raw_cpu_buffers;
                                            debug=_GAUGE_COPY_DEBUG[], label="next_U ")
                end
            else
                @timeit to "jld2_fetch"    _next_U = fetch(cpu_prefetch)
                @timeit to "substitute_U!" begin
                    for mu in eachindex(U_template)
                        _fast_substitute_U!(U_template[mu], _next_U[mu], _staging[mu];
                                            debug=_GAUGE_COPY_DEBUG[], label="U[$mu] ")
                    end
                    # _sanitize_U!(U_template)
                end
            end
        end

        _mem_after_gauge = _MEM_DEBUG[] ? _mem_snapshot() : UInt(0)
        println("  epoch $epoch / $epochs  loss=$(real(sum(temp)))")
        if _MEM_DEBUG[]
            _to_mb(x) = round(x / 1048576, digits=1)
            println("  [MEM] fwd+bwd:+$(_to_mb(_mem_after_bwd-_mem_epoch_start)) MB  opt:+$(_to_mb(_mem_after_opt-_mem_after_bwd)) MB  gauge:+$(_to_mb(_mem_after_gauge-_mem_after_opt)) MB  | epoch:+$(_to_mb(_mem_after_gauge-_mem_epoch_start)) MB  (abs:$(round(_mem_after_gauge/1048576,digits=0)) MB)")
        end
    end

    println("\n===== forward timers =====")
    sync_layers!(layers)
    show(to["forward"]; sortby=:name)
    println("\n===== backward timers =====")
    show(to["backward"]; sortby=:name)
    println("\n===== total timers =====")
    show(to; sortby=:time, compact=true)
    println()
    return to
end
