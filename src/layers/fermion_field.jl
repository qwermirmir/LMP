# fermionfield_Layer: the trainable input fermion fields of the model.

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

_has_gpu(layer::fermionfield_Layer) = !isempty(layer.x) && layer.x[1] isa WF3x1GPU

# --- fermionfield_Layer ---
function _to_cpu(layer::fermionfield_Layer)
    layer.x = _to_cpu_fermions(layer.x)
    return layer
end

# --- fermionfield_Layer ---
function _to_gpu(layer::fermionfield_Layer, blocks=nothing)
    layer.x = _to_gpu_fermions(layer.x; blocks=blocks)
    return layer
end

function (layer::fermionfield_Layer)(x::Any)
    return layer.x
end

"""
    refresh_fermionfield!(layer::fermionfield_Layer, U)

Refresh stale CuArray fermion fields loaded from JLD2.
Currently this is a no-op because WilsonFermion data does not own U.
"""
function refresh_fermionfield!(layer::fermionfield_Layer, U)
    # WilsonFermion does not store U — gauge field is passed at apply-time.
    # No reallocation needed when U changes; existing spinor data remains valid.
    return layer
end

_refresh_layer!(layer::fermionfield_Layer, D_new, U_new) = refresh_fermionfield!(layer, U_new)

function gradient(layer::fermionfield_Layer, ::Identity, z::Any, d::Vector{T} ,  γ::Vector{Spin_matrix} ) where T<:WF3x1

    return d
    
end

function DeltaRule(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1)
    
end

function DeltaRule!(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1, delta::Any)
    
end

# function update!(layer::L2Norm_Layer, opt::Adam, grad::Any)

  
# end


function update!(layer::fermionfield_Layer, opt::Adam, grad::Any)
    param = layer.x

    if length(param) ==1
        wf_adam_step!(param[1], opt , grad[1] )
    
    else 
        for i in 1:length(param)
            wf_adam_step!(param[i], opt , grad[i] )
        end
    end
    # wf_adam_step!(layer.x, opt, grad[1][1])

    return nothing
end
