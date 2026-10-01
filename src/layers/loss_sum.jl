# LossSumLayer: a weighted sum of two loss layers applied to the same input.

"""
    LossSumLayer(l1, l2, w1=1.0, w2=1.0, activation=Identity())

Loss layer `loss = w1 * l1(x) + w2 * l2(x)`. Its delta is the same weighted
sum of the deltas of `l1` and `l2`. Either may itself be a `LossSumLayer`.
"""
mutable struct LossSumLayer{L1<:Layer,L2<:Layer} <: Layer
    l1::L1
    l2::L2
    w1::Float64
    w2::Float64
    activation::ActivationFunction

    function LossSumLayer(l1::L1, l2::L2,
                          w1::Real=1.0, w2::Real=1.0,
                          activation::ActivationFunction=Identity()) where {L1<:Layer,L2<:Layer}
        new{L1,L2}(l1, l2, Float64(w1), Float64(w2), activation)
    end
end

# ---------------------------------------------------------------------------
# Forward and backward passes
# ---------------------------------------------------------------------------

function (L::LossSumLayer)(x::AbstractVector{<:WF3x1})
    return L.w1 * L.l1(x) + L.w2 * L.l2(x)
end

function gradient(L::LossSumLayer, ::ActivationFunction, z, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end

# The delta of a weighted sum of losses is the same weighted sum of their deltas.
function DeltaRule(L::LossSumLayer,  ::Identity, z::Vector{<:WF3x1}, batch::Int)
    d1 = DeltaRule(L.l1, L.l1.activation, z, batch)
    d2 = DeltaRule(L.l2, L.l2.activation, z, batch)
    @assert length(d1) == length(d2) "DeltaRule outputs from loss layers must have the same length."
    result = similar(d1)
    for i in eachindex(d1)
        result[i] = similar(d1[i])
        axpby!(L.w1, d1[i], 0, result[i])
        axpby!(L.w2, d2[i], 1, result[i])
    end
    return result
end

function DeltaRule!(L::LossSumLayer, ::Identity, z::Vector{<:WF3x1}, delta::Vector, batch::Int)
    d = DeltaRule(L, L.activation, z, batch)
    @assert length(delta) == length(d) "DeltaRule! output buffer has wrong length."
    for i in eachindex(d)
        axpby!(1, d[i], 0, delta[i])
    end
    return delta
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion and refresh, delegated to the two losses
# ---------------------------------------------------------------------------

_has_gpu(layer::LossSumLayer) = _has_gpu(layer.l1) || _has_gpu(layer.l2)

function _to_cpu(layer::LossSumLayer)
    l1_cpu = _to_cpu(layer.l1)
    l2_cpu = _to_cpu(layer.l2)
    return LossSumLayer(l1_cpu, l2_cpu, layer.w1, layer.w2, layer.activation)
end

function _to_gpu(layer::LossSumLayer, blocks=nothing)
    l1_gpu = _to_gpu(layer.l1, blocks)
    l2_gpu = _to_gpu(layer.l2, blocks)
    return LossSumLayer(l1_gpu, l2_gpu, layer.w1, layer.w2, layer.activation)
end

function _to_gpu(layer::LossSumLayer, blocks, shared_U)
    l1_gpu = _to_gpu(layer.l1, blocks, shared_U)
    l2_gpu = _to_gpu(layer.l2, blocks, shared_U)
    return LossSumLayer(l1_gpu, l2_gpu, layer.w1, layer.w2, layer.activation)
end

function _refresh_layer!(layer::LossSumLayer, D_new, U_new)
    _refresh_layer!(layer.l1, D_new, U_new)
    _refresh_layer!(layer.l2, D_new, U_new)
end

function _refresh_forward_error!(layer::LossSumLayer, D_new)
    _refresh_forward_error!(layer.l1, D_new)
    _refresh_forward_error!(layer.l2, D_new)
    return nothing
end
