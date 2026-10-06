# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/activation.jl.

mutable struct GramSchmidt_Layer <: Layer
    x::Vector{WF3x1}           # forward에서 직교화 결과 캐시 (backward에서 재사용)
    activation::ActivationFunction
    function GramSchmidt_Layer(activation::ActivationFunction= Identity())
        new(WF3x1[], activation)
    end 
end

_has_gpu(layer::GramSchmidt_Layer) = !isempty(layer.x) && layer.x[1] isa WF3x1GPU

# --- GramSchmidt_Layer ---
function _to_cpu(layer::GramSchmidt_Layer)
    layer.x = _to_cpu_fermions(layer.x)
    return layer
end

# --- GramSchmidt_Layer ---
function _to_gpu(layer::GramSchmidt_Layer, blocks=nothing)
    layer.x = _to_gpu_fermions(layer.x; blocks=blocks)
    return layer
end

function _refresh_layer!(layer::GramSchmidt_Layer, D_new, U_new)
    for i in eachindex(layer.x)
        gauss_distribution_fermion!(layer.x[i])
    end
end

function (layer::GramSchmidt_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)

    # Reuse allocated Gram-Schmidt fields when possible.
    if length(layer.x) != l
        layer.x = [deepcopy(x[i]) for i in 1:l]
    end
    q = layer.x

    # Modified Gram-Schmidt.
    copyto!(q[1].f, x[1].f)
    @inbounds for i in 2:l
        copyto!(q[i].f, x[i].f)
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], q[i]) / denom
            axpby!(-coeff, q[j], 1, q[i])   # q[i] -= proj_{q[j]}(q[i])
        end
    end

    return q
end

function gradient(layer::GramSchmidt_Layer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    return nothing
end 

function DeltaRule(layer::GramSchmidt_Layer, ::Identity,
                   z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1

    q = layer.x  # forward에서 캐시된 직교화 벡터
    l = length(d)

    result = [deepcopy(d[i]) for i in 1:l]

    @inbounds for i in 1:l
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], result[i]) / denom
            axpby!(-coeff, q[j], 1, result[i])
        end
    end

    return result
end

function DeltaRule!(layer::GramSchmidt_Layer, ::Identity,
                   z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    q = layer.x  # forward에서 캐시된 직교화 벡터
    l = length(d)

    @inbounds for i in 1:l
        copyto!(delta[i].f, d[i].f)
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], delta[i]) / denom
            axpby!(-coeff, q[j], 1, delta[i])
        end
    end

end
