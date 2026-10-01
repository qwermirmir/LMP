# fermionfield_Layer: the trainable fermion fields at the start of a model.

"""
    fermionfield_Layer(x, activation=Identity())
    fermionfield_Layer(x, In, activation=Identity())

First layer of a model. It ignores its input and returns its own fields
`layer.x`, which are trained like weights. The second form creates `In`
Gaussian random fields shaped like `x`.

Its gradient is the delta arriving from the next layer, `dL/dx = d`, and the
Adam update acts on the fields directly (`wf_adam_step!`).
"""
mutable struct fermionfield_Layer <: Layer
    x::Vector{WF3x1}
    In::Int
    activation::ActivationFunction
    function fermionfield_Layer(x::WF3x1, activation::ActivationFunction= Identity())
        i=1
        new([x], i, activation)
    end
    function fermionfield_Layer(x::WF3x1, In::Int, activation::ActivationFunction= Identity())
        xs=Vector{WF3x1}(undef, In)
        for i in 1:In
            xs[i]=deepcopy(x)
            gauss_distribution_fermion!(xs[i])
        end
        new(xs, In, activation)
    end
end

# ---------------------------------------------------------------------------
# Forward and backward passes
# ---------------------------------------------------------------------------

function (layer::fermionfield_Layer)(x::Any)
    return layer.x
end

# Nothing comes before this layer, so there is no delta to propagate.
function DeltaRule(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1)
end

function DeltaRule!(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1, delta::Any)
end

# ---------------------------------------------------------------------------
# Gradient and Adam update
# ---------------------------------------------------------------------------

function gradient(layer::fermionfield_Layer, ::Identity, z::Any, d::Vector{T} ,  γ::Vector{Spin_matrix} ) where T<:WF3x1
    return d
end

function update!(layer::fermionfield_Layer, opt::Adam, grad::Any)
    param = layer.x

    if length(param) ==1
        wf_adam_step!(param[1], opt , grad[1] )
    else
        for i in 1:length(param)
            wf_adam_step!(param[i], opt , grad[i] )
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion
# ---------------------------------------------------------------------------

_has_gpu(layer::fermionfield_Layer) = !isempty(layer.x) && layer.x[1] isa WF3x1GPU

function _to_cpu(layer::fermionfield_Layer)
    layer.x = _to_cpu_fermions(layer.x)
    return layer
end

function _to_gpu(layer::fermionfield_Layer, blocks=nothing)
    layer.x = _to_gpu_fermions(layer.x; blocks=blocks)
    return layer
end

"""
    refresh_fermionfield!(layer::fermionfield_Layer, U)

Does nothing: Wilson fermion fields do not hold the gauge field, so they stay
valid when `U` changes. Kept so that `refresh_model!` can treat every layer
alike.
"""
function refresh_fermionfield!(layer::fermionfield_Layer, U)
    return layer
end

_refresh_layer!(layer::fermionfield_Layer, D_new, U_new) = refresh_fermionfield!(layer, U_new)
