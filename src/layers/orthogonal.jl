# Orthogonal_Layer: penalty on the overlaps between the input fields.

"""
    Orthogonal_Layer(coeff, activation=Identity())

Loss layer that pushes the input fields towards mutual orthogonality:

    loss = coeff * sum_{i > j} |<x[j], x[i]>|^2

Backward pass, with `c = <z[j], z[i]>` for every pair `i > j`:

    delta[i] += 2 coeff conj(c) conj(z[j]),    delta[j] += 2 coeff c conj(z[i])

Only the allocating `DeltaRule` exists; combine it with another loss through
`LossSumLayer`.
"""
struct Orthogonal_Layer <: Layer
    coeff::Float64
    activation::ActivationFunction
    function Orthogonal_Layer(coeff::Float64, activation::ActivationFunction= Identity())
        new(coeff, activation)
    end
end

function (layer::Orthogonal_Layer)(x::AbstractVector{<:WF3x1})

    l=length(x)
    a=layer.coeff

    result=0.0

    for i in 2:l
        for j in 1:i-1
            c=inner_product(x[j],x[i])
            result+=c'*c
        end
    end

    return a*result
end

function DeltaRule(layer::Orthogonal_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    result = Vector{WF3x1}(undef, length(z))
    temp = similar(z[1])
    for i in 1:length(z)
        result[i]=deepcopy(temp)
    end

    a=layer.coeff
    l=length(z)

    for i in 2:l
        for j in 1:i-1
            c=inner_product(z[j],z[i])
            result[i]+=a*c'*conj(z[j])
            result[j]+=a*c*conj(z[i])
        end
    end

    return 2*result
end

# A loss layer has no weights.
function gradient(layer:: Orthogonal_Layer , ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})
end
