# Forward and backward passes through a model, which is a vector of layers
# whose last layer is a loss. Activations are kept in a vector u:
#
#     u[1] = [input],   u[i+1] = layers[i](u[i]),   loss = layers[L](u[L])
#
# and deltas in a vector delta with delta[k] = dL/du[k+1]. Each u[k] and
# delta[k] is itself a vector of fermion fields.

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

"""
    forward_propagation(layers, input, target) -> (u, loss)

Runs the model and returns the activations `u` and the loss. If the last layer
has a `target` field it is set to `target` first. The standard models start
with a `fermionfield_Layer`, which ignores `input`.
"""
function forward_propagation(layers::Vector{T}, input::Any, target::Any) where {T<:Layer}
    L = length(layers)

    let last = layers[L]
        if hasfield(typeof(last), :target)
            last.target = target
        end
    end

    u = Vector{Any}(undef, L)
    u[1] = [input]

    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    f = layers[L](u[L])

    return u, f
end

"""
    (layers::Vector{<:Layer})(x)

Calling a model applies its layers in turn without keeping the activations,
so `layers([U])` returns the loss. The training scripts use it to print the
final loss.
"""
function (layers::Vector{T})(x::Any) where T<:Layer
    result = x
    for layer in layers
        result = layer(result)
    end
    return result
end

"""
    forward_propagation!(layers, input, target, u) -> loss

Same as `forward_propagation`, writing the activations into an existing `u`.
"""
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

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

"""
    backward_propagation!(layers, u, batch, delta)

Fills `delta[k] = dL/du[k+1]` for `k = 1..L-1`, from the activations of a
forward pass. Weight gradients are computed afterwards with
`gradient(layers[i], layers[i].activation, u[i], delta[i], gamma)`.
"""
function backward_propagation!(layers::Vector{T}, u::Vector{Any}, batch::Int, delta::Vector{Any}) where T<:Layer
    L = length(layers)

    @assert length(u) == L
    @assert length(delta) == L-1

    # The loss layer maps u[L] to the loss.
    DeltaRule!(layers[L], layers[L].activation, u[L], delta[L-1], batch)

    for j in (L-1):-1:2
        DeltaRule!(
            layers[j],
            layers[j].activation,
            u[j],
            delta[j],   # dL/du[j+1]
            delta[j-1]  # dL/du[j]
        )
    end

end

"""
    backward_and_gradient!(layers, u, batch, delta_tail, buf_a, buf_b,
                           transition_j, d, gamma, delta1_buf=nothing)

The backward pass used by the training loop. It computes each layer's weight
gradient as soon as its delta is known and adds it to `d[j]`, and it stores
deltas in few buffers:

- `delta_tail[j]` for the layers from `transition_j` to `L-1`, whose fields
  differ in number from the body of the model;
- `buf_a` and `buf_b`, used alternately for the body layers below
  `transition_j`, which all have the same shape;
- `delta1_buf` for the delta at `u[2]`, needed by the gradient of layer 1.

`train` builds these buffers once from a first forward pass.
"""
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
    delta1_buf=nothing
) where T<:Layer

    L = length(layers)

    # 1. Loss layer.
    DeltaRule!(layers[L], layers[L].activation, u[L], delta_tail[L-1], batch)

    # 2. Tail layers, j = L-1 down to transition_j.
    for j in (L-1):-1:transition_j
        cur = delta_tail[j]

        gi = gradient(layers[j], layers[j].activation, u[j], cur, gamma)
        _accum_gradient!(d, j, gi)

        if j > transition_j
            DeltaRule!(layers[j], layers[j].activation, u[j], cur, delta_tail[j-1])
        else
            # Hand over to the body buffers.
            DeltaRule!(layers[j], layers[j].activation, u[j], cur, buf_a)
        end
    end

    # 3. Body layers, j = transition_j-1 down to 2, alternating buf_a and buf_b.
    cur_buf, nxt_buf = buf_a, buf_b
    for j in (transition_j - 1):-1:2
        gi = gradient(layers[j], layers[j].activation, u[j], cur_buf, gamma)
        _accum_gradient!(d, j, gi)

        if j > 2
            DeltaRule!(layers[j], layers[j].activation, u[j], cur_buf, nxt_buf)
            cur_buf, nxt_buf = nxt_buf, cur_buf
        end
    end

    # 4. Layer 1, whose input u[1] has a different shape.
    if delta1_buf !== nothing && L >= 2
        DeltaRule!(layers[2], layers[2].activation, u[2], cur_buf, delta1_buf)
        gi1 = gradient(layers[1], layers[1].activation, u[1], delta1_buf, gamma)
        _accum_gradient!(d, 1, gi1)
    end
end

# Builds the delta buffers of backward_and_gradient! from one forward pass.
# The body of the model is the run of layers whose outputs have as many
# fields as u[4]; the tail is everything from the first layer after the body
# to the loss.
function _backward_buffer_setup!(layers, D, depth::Int)
    u, _ = forward_propagation(layers, D, D)

    body_size = length(u[min(4, depth)])

    # Tail layers get their own delta buffers.
    delta_tail = Dict{Int,Any}(depth-1 => deepcopy(u[depth]))
    for j in 3:depth-2
        length(u[j+1]) != body_size && (delta_tail[j] = deepcopy(u[j+1]))
    end
    transition_j = minimum(keys(delta_tail))
    for j in transition_j:depth-1
        haskey(delta_tail, j) || (delta_tail[j] = deepcopy(u[j+1]))
    end

    # Two alternating body buffers shaped like u[transition_j].
    buf_a = deepcopy(u[transition_j])
    buf_b = deepcopy(u[transition_j])

    # u[1] is [D], so the delta at u[2] needs its own buffer.
    delta1_buf = depth >= 2 ? deepcopy(u[2]) : nothing

    return u, delta_tail, transition_j, buf_a, buf_b, delta1_buf
end

# Adds a layer gradient into the accumulator d[j]. Some layers return their own
# cached gradient buffer, so with batch > 1 the first sample's gradient can be
# overwritten by the next one (see CLEANUP_PLAN.md).
function _accum_gradient!(d::Vector{Any}, j::Int, gi)
    gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)
    return nothing
end

# ---------------------------------------------------------------------------
# Activation checkpointing
#
# For large models the activations of every layer do not fit in GPU memory.
# With checkpoint interval k only every k-th activation is kept; the backward
# pass recomputes each segment between two kept activations from the first
# one.
# ---------------------------------------------------------------------------

function _is_checkpoint_activation(idx::Int, depth::Int, interval::Int)
    interval <= 0 && return true
    return idx == 1 || idx == depth || ((idx - 1) % interval == 0)
end

# Frees the forward buffers of a layer so that only the kept activations
# remain in memory.
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

function _clear_forward_caches!(layers, first_layer::Int=1, last_layer::Int=length(layers)-1)
    for i in max(first_layer, 1):min(last_layer, length(layers)-1)
        _clear_layer_forward_cache!(layers[i])
    end
    return nothing
end

# Forward pass that keeps only the checkpointed activations; returns them by
# index and stores the loss in temp[b].
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

# Recomputes the activations from u[start_idx] to u[stop_idx].
function _recompute_checkpoint_segment!(to, layers, u::Vector{Any}, checkpoints,
                                        start_idx::Int, stop_idx::Int)
    fill!(u, nothing)
    u[start_idx] = checkpoints[start_idx]
    for i in start_idx:(stop_idx-1)
        @timeit to "checkpoint_recompute" u[i+1] = layers[i](u[i])
    end
    return nothing
end

# Builds the delta buffers of backward_and_gradient! from one forward pass,
# freeing the activations as it goes.
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

# backward_and_gradient! over the checkpointed segments, last segment first,
# recomputing each segment's activations before its backward pass.
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
