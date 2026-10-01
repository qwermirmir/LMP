# Forward and backward passes through a list of layers, including the
# activation-checkpointed variants used for large models.

function forward_propagation(layers::Vector{T}, input::Any, target::Any) where {T<:Layer}
    L = length(layers)

    # u[1] is input, u[2] is layer 1 output, ..., u[L] is layer L-1 output.
    let last = layers[L]
        if hasfield(typeof(last), :target)
            last.target = target             # (= setproperty!(last, :target, target))
        end
    end

    u = Vector{Any}(undef, L)
    u[1] = [input]

    # Forward through layers 1..L-1.
    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    # Final layer returns the scalar loss.
    f = layers[L](u[L])

    return u, f
end

# Inplace - u
function forward_propagation!(layers::Vector{T}, input::Any, target::Any, u::Vector{Any}) where {T<:Layer}
    L = length(layers)
    @assert length(u) == L "u length must match layers."

    let last = layers[L]
        if hasfield(typeof(last), :target)
            last.target = target
        end
    end

    u[1] = [input]

    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    f = layers[L](u[L])

    return f
end

function backward_propagation!(layers::Vector{T}, u::Vector{Any}, batch::Int, delta::Vector{Any}) where T<:Layer
    L = length(layers)

    @assert length(u) == L
    @assert length(delta) == L-1

    # delta[k] corresponds to d(loss) / d(u[k+1]).
    # Final layer maps u[L] to the scalar loss.
    DeltaRule!(layers[L], layers[L].activation, u[L], delta[L-1], batch)

    # hidden layers: j = L-1, ..., 2
    for j in (L-1):-1:2
        DeltaRule!(
            layers[j],
            layers[j].activation,
            u[j],
            delta[j],   # d(loss) / d(u[j+1])
            delta[j-1]  # d(loss) / d(u[j])
        )
    end

end

# Replace NaN/Inf values with zero using a GPU-native in-place map.

# Backward + gradient with ping-pong buffers.
# delta_tail stores tail deltas that do not fit the body ping-pong buffers.
# buf_a/buf_b are the two body ping-pong buffers.
# transition_j is the first tail layer before entering body ping-pong.
function backward_and_gradient!(
    layers::Vector{T},
    u::Vector{Any},
    batch::Int,
    delta_tail::Dict{Int,Any},
    buf_a::Vector,
    buf_b::Vector,
    transition_j::Int,
    d::Vector{Any},
    gamma,
    delta1_buf=nothing # delta buffer for layer 1 gradient against u[1]
) where T<:Layer

    L = length(layers)

    # 1. Loss layer: write delta[L-1] into delta_tail[L-1].
    DeltaRule!(layers[L], layers[L].activation, u[L], delta_tail[L-1], batch)

    # 2. Tail: j = L-1 down to transition_j.
    for j in (L-1):-1:transition_j
        cur = delta_tail[j]

        gi = gradient(layers[j], layers[j].activation, u[j], cur, gamma)
        gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)

        if j > transition_j
            DeltaRule!(layers[j], layers[j].activation, u[j], cur, delta_tail[j-1])
        else
            # j == transition_j: hand off into body ping-pong.
            DeltaRule!(layers[j], layers[j].activation, u[j], cur, buf_a)
        end
    end

    # 3. Body ping-pong: j = transition_j-1 down to 2.
    # cur_buf holds delta[j], nxt_buf receives the next step.
    cur_buf, nxt_buf = buf_a, buf_b
    for j in (transition_j - 1):-1:2
        gi = gradient(layers[j], layers[j].activation, u[j], cur_buf, gamma)
        gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)

        if j > 2
            DeltaRule!(layers[j], layers[j].activation, u[j], cur_buf, nxt_buf)
            cur_buf, nxt_buf = nxt_buf, cur_buf
        end
        # j == 2: layer 1 DeltaRule! is handled below.
    end

    # 4. Layer 1: use delta1_buf because u[1] has a different shape.
    if delta1_buf !== nothing && L >= 2
        DeltaRule!(layers[2], layers[2].activation, u[2], cur_buf, delta1_buf)
        gi1 = gradient(layers[1], layers[1].activation, u[1], delta1_buf, gamma)
        gi1 !== nothing && (d[1] = d[1] === nothing ? gi1 : d[1] + gi1)
    end
end

# Lazy-loading variant: gauge_paths JLD2
# U_template is allocated on GPU once.
# JLD2 @async O
function _is_checkpoint_activation(idx::Int, depth::Int, interval::Int)
    interval <= 0 && return true
    return idx == 1 || idx == depth || ((idx - 1) % interval == 0)
end

function _clear_layer_forward_cache!(layer)
    if layer isa DenseLinearLayer
        layer._result = WF3x1[]
        layer._yi = WF3x1[]
        layer._dl_init = false
        layer._input_all = nothing
        layer._result_all = nothing
        layer._d_buf = nothing
        layer._ci_all = nothing
    elseif layer isa PTLayer
        layer._hopped = WF3x1[]
        layer._result = WF3x1[]
        layer._initialized = false
        layer._hopped_all = nothing
    elseif layer isa NormLayer
        layer._result = WF3x1[]
        layer._norm_init = false
    elseif layer isa BlockScalar_Layer
        layer._result = WF3x1[]
        layer._blocks = Vector{WF3x1}[]
        layer._initialized = false
    end
    return nothing
end

function _checkpointed_forward_loss!(to, fwd_names, layers, D, u::Vector{Any},
                                     temp::Vector{ComplexF64}, b::Int,
                                     depth::Int, checkpoint_interval::Int)
    checkpoints = Dict{Int,Any}(1 => [D])
    fill!(u, nothing)
    u[1] = checkpoints[1]

    for i in 1:(depth-1)
        @timeit to fwd_names[i] u[i+1] = layers[i](u[i])
        if _is_checkpoint_activation(i + 1, depth, checkpoint_interval)
            checkpoints[i+1] = deepcopy(u[i+1])
        end
        if i > 1
            _clear_layer_forward_cache!(layers[i-1])
            u[i] = nothing
        end
    end

    if layers[depth] isa Ritz_Layer && !isempty(u[depth]) && u[depth][1] isa WF3x1GPU
        CUDA.synchronize()
        GC.gc(false)
    end
    @timeit to fwd_names[depth] temp[b] = layers[depth](u[depth])
    _clear_layer_forward_cache!(layers[depth-1])
    u[depth] = nothing
    return checkpoints
end

function _checkpointed_buffer_setup!(layers, D, depth::Int)
    u = Vector{Any}(undef, depth)
    fill!(u, nothing)
    u[1] = [D]

    body_idx = min(4, depth)
    body_size = nothing
    delta_tail = Dict{Int,Any}()
    transition_j = nothing
    transition_input = nothing
    delta1_buf = nothing

    for i in 1:(depth-1)
        u[i+1] = layers[i](u[i])

        if i + 1 == 2
            delta1_buf = deepcopy(u[2])
        end
        if i + 1 == body_idx
            body_size = length(u[i+1])
        end

        if body_size !== nothing
            if 3 <= i <= depth - 2 && length(u[i+1]) != body_size
                if transition_j === nothing
                    transition_j = i
                    transition_input = deepcopy(u[i])
                end
            end
            if transition_j !== nothing && transition_j <= i <= depth - 1
                delta_tail[i] = deepcopy(u[i+1])
            end
        end

        if i > 1
            _clear_layer_forward_cache!(layers[i-1])
            u[i] = nothing
        end
    end

    if transition_j === nothing
        transition_j = depth - 1
        transition_input = deepcopy(u[depth-1])
        delta_tail[depth-1] = deepcopy(u[depth])
    elseif !haskey(delta_tail, depth - 1)
        delta_tail[depth-1] = deepcopy(u[depth])
    end

    buf_a = deepcopy(transition_input)
    buf_b = deepcopy(transition_input)

    _clear_layer_forward_cache!(layers[depth-1])
    fill!(u, nothing)
    _force_cuda_reclaim!()

    return u, delta_tail, transition_j, buf_a, buf_b, delta1_buf
end

function _checkpointed_backward_timed!(to, bwd_names, layers, u::Vector{Any}, checkpoints,
                                       batch::Int, delta_tail, buf_a, buf_b,
                                       transition_j::Int, d::Vector{Any}, gamma,
                                       delta1_buf)
    L = length(layers)
    bounds = sort!(collect(keys(checkpoints)))
    cur_buf, nxt_buf = buf_a, buf_b

    for seg in (length(bounds)-1):-1:1
        start_idx = bounds[seg]
        stop_idx = bounds[seg+1]
        _recompute_checkpoint_segment!(to, layers, u, checkpoints, start_idx, stop_idx)

        if stop_idx == L
            @timeit to bwd_names[L] DeltaRule!(layers[L], layers[L].activation, u[L], delta_tail[L-1], batch)
        end

        tail_hi = min(stop_idx - 1, L - 1)
        tail_lo = max(start_idx, transition_j)
        if tail_hi >= tail_lo
            for j in tail_hi:-1:tail_lo
                cur = delta_tail[j]
                @timeit to bwd_names[j] begin
                    gi = gradient(layers[j], layers[j].activation, u[j], cur, gamma)
                    _accum_gradient!(d, j, gi)
                    if j > transition_j
                        DeltaRule!(layers[j], layers[j].activation, u[j], cur, delta_tail[j-1])
                    else
                        DeltaRule!(layers[j], layers[j].activation, u[j], cur, buf_a)
                    end
                end
            end
        end

        body_hi = min(stop_idx - 1, transition_j - 1)
        body_lo = max(start_idx, 2)
        if body_hi >= body_lo
            for j in body_hi:-1:body_lo
                @timeit to bwd_names[j] begin
                    gi = gradient(layers[j], layers[j].activation, u[j], cur_buf, gamma)
                    _accum_gradient!(d, j, gi)
                    if j > 2
                        DeltaRule!(layers[j], layers[j].activation, u[j], cur_buf, nxt_buf)
                        cur_buf, nxt_buf = nxt_buf, cur_buf
                    end
                end
            end
        end

        if start_idx == 1 && delta1_buf !== nothing && L >= 2
            @timeit to bwd_names[2] DeltaRule!(layers[2], layers[2].activation, u[2], cur_buf, delta1_buf)
            @timeit to bwd_names[1] begin
                gi1 = gradient(layers[1], layers[1].activation, u[1], delta1_buf, gamma)
                _accum_gradient!(d, 1, gi1)
            end
        end

        _clear_forward_caches!(layers, start_idx, stop_idx - 1)
        fill!(u, nothing)
    end

    return d
end
