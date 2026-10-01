# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_types.jl.

struct Softmax <: ActivationFunction end

struct RealSoftmax <: ActivationFunction end

struct ModRelu <: ActivationFunction end

function (s::Softmax)(x::Spin_matrix)
    A = x.sm
    maxA = maximum(abs.(A), dims=2)      # 각 행의 복소수 절댓값 최대
    expA = exp.(A .- maxA)               # 안정화된 지수
    row_sum = sum(expA, dims=2)
    return Spin_matrix(expA ./ row_sum)
end

function (s::Softmax)(x::Vector{Spin_matrix})
    y = similar(x)

    for i in 1:length(x)
        A = x[i].sm
        maxA = maximum(abs.(A), dims=2)
        expA = exp.(A .- maxA)
        row_sum = sum(expA, dims=2)
        y[i] = Spin_matrix(expA ./ row_sum)
    end

    return y
end

function (s::Softmax)(x::Spin_matrix_GPU)
    A= x.sm
    maxA = maximum(abs.(A), dims=2)      # 각 행의 복소수 절댓값 최대
    expA = exp.(A .- maxA)               # 안정화된 지수
    row_sum = sum(expA, dims=2)
    return Spin_matrix_GPU(expA ./ row_sum)
end

function (s::Softmax)(x::Vector{Spin_matrix_GPU})
    y = similar(x)

    for i in 1:length(x)
        A = x[i].sm
        maxA = maximum(abs.(A), dims=2)
        expA = exp.(A .- maxA)
        row_sum = sum(expA, dims=2)
        y[i] = Spin_matrix_GPU(expA ./ row_sum)
    end

    return y
end

function (r::ModRelu)(x::Spin_matrix)
    A = x.sm
    modA = abs.(A)
    phaseA = angle.(A)
    activated_modA = max.(modA .-0.1, 0.0)  # ReLU 적용
    activated_A = activated_modA .* exp.(im .* phaseA)
    return Spin_matrix(activated_A)
end

# function (r::ModRelu)(x::Spin_matrix)
#     A = x.sm
#     for i in 1:4
#         if abs(A[i,i])>0.1
#             A[i,i]= (abs(A[i,i])-50.0)*A[i,i]/abs(A[i,i])
#         else
#             A[i,i]=0
#         end
#     end

#     return Spin_matrix(A)
# end







function (r::ModRelu)(xs::Vector{Spin_matrix})
    ys = similar(xs)
    @inbounds for i in eachindex(xs)
        ys[i] = r(xs[i])
    end
    return ys
end

struct Decoder <: Layer
    activation::ActivationFunction
    function Decoder(activation::ActivationFunction )
        new(activation)
    end 
end 
