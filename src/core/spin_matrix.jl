# 4x4 complex matrices acting on the spinor index (Spin_matrix on the CPU,
# Spin_matrix_GPU on the GPU), their arithmetic, the Clifford basis, weight
# initialisation, and packing of weight vectors to and from the GPU.

mutable struct Spin_matrix <: AbstractSpinMatrix
    sm::Matrix{ComplexF64}
end

size(S::Spin_matrix) = size(S.sm)  

length(S::Spin_matrix) = length(S.sm)  

ndims(S::Spin_matrix) = 2

# Overload Base.adjoint for Spin_matrix
Base.adjoint(S::Spin_matrix) = Spin_matrix(adjoint(S.sm))

broadcastable(S::Spin_matrix) = S.sm  # broadcast over the underlying matrix

function Base.adjoint(S::Vector{Spin_matrix})
    return [adjoint(s) for s in S]
end

function Random_SpinM(matrix_type::String = "complex")
    if matrix_type == "complex"
        mat = randn(4,4) .+ im*randn(4,4)
    elseif matrix_type == "real"
        mat = randn(4,4)
    elseif matrix_type=="zero"
        mat = zeros(ComplexF64, 4,4)

    elseif matrix_type=="Init"
        mat = Matrix{ComplexF64}(I, 4, 4)
    
    else
        error("unsupported matrix_type; use 'complex' or 'real'.")
    end
    
    return Spin_matrix(mat)
end

function Base.getindex(A::Spin_matrix, I...)
    return A.sm[I...]
end

function Base.setindex!(A::Spin_matrix, v, I...)
    A.sm[I...] = v
    return A
end

function Base.:*(a::Spin_matrix,b::Spin_matrix)

    c=deepcopy(b)

    c.sm=a.sm*b.sm

    return c
end

function Base.:*(a::Vector{Spin_matrix},b::Vector{Spin_matrix})

    c=deepcopy(b)


    for i in 1:length(b)
        c[i]=a[i]*b[i]
    end

    return c
end

function Base.:*(c::Number,a::Spin_matrix)

    b=deepcopy(a)

    b.sm=c*b.sm

    return b
end

function Base.:+(a::Spin_matrix,b::Spin_matrix)

    c=deepcopy(b)

    c.sm=a.sm+b.sm

    return c
end

function Base.:-(a::Spin_matrix,b::Spin_matrix)

    c=deepcopy(b)

    c.sm=a.sm-b.sm

    return c
end

function Base.:/(a::Spin_matrix,c::Number)

    b=deepcopy(a)

    b.sm=a.sm/c

    return b
end

function Base.:/(a::Vector{Spin_matrix},c::Number)

    b=deepcopy(a)

    for i in 1:length(a)
        b[i].sm=a[i].sm/c
    end

    return b
end

function Base.:*(a::Vector{Spin_matrix},c::Number)

    b=deepcopy(a)

    for i in 1:length(a)
        b[i].sm=a[i].sm*c
    end

    return b
end

Base.transpose(S::Spin_matrix) = Spin_matrix(transpose(S.sm))

Base.similar(A::Spin_matrix, ::Type{ComplexF64}, dims::Dims{2}) = Spin_matrix(zeros(ComplexF64, dims))

function Base.similar(A::Vector{Spin_matrix})
    return [Spin_matrix(zeros(ComplexF64, size(a))) for a in A]
end

function Clifford()
    basis = Vector{Spin_matrix}()

    # 1) the identity
    id4 = Random_SpinM("zero")
    for i in 1:4
        id4[i,i] = 1
    end
    push!(basis, id4)

    # 2) gamma1
    gamma1 = Random_SpinM("zero")
    gamma1[1,3] = 1
    gamma1[2,4] = 1
    gamma1[3,1] = 1
    gamma1[4,2] = 1
    push!(basis, gamma1)

    # 3) gamma2
    gamma2 = Random_SpinM("zero")
    gamma2[1,4] = -im
    gamma2[2,3] =  im
    gamma2[3,2] =  im
    gamma2[4,1] = -im
    push!(basis, gamma2)

    # 4) gamma3
    gamma3 = Random_SpinM("zero")
    gamma3[1,3] =  1
    gamma3[2,4] = -1
    gamma3[3,1] = -1
    gamma3[4,2] =  1
    push!(basis, gamma3)

    # 5) gamma4
    gamma4 = Random_SpinM("zero")
    gamma4[1,2] = 1
    gamma4[2,1] = 1
    gamma4[3,4] = -1
    gamma4[4,3] = -1
    push!(basis, gamma4)

    # 6) gamma5 = diag(-1, -1, +1, +1)
    gamma5 = Random_SpinM("zero")
    gamma5[1,1] = -1
    gamma5[2,2] = -1
    gamma5[3,3] =  1
    gamma5[4,4] =  1
    push!(basis, gamma5)

    # 7) pairwise products: gamma^mu gamma^nu (mu < nu)

    push!(basis,gamma1*gamma2)
    push!(basis,gamma1*gamma3)
    push!(basis,gamma1*gamma4)
    push!(basis,gamma2*gamma3)
    push!(basis,gamma2*gamma4)
    push!(basis,gamma3*gamma4)

    # 8) gamma5 gamma_mu, four matrices
    push!(basis, gamma5*gamma1)
    push!(basis, gamma5*gamma2)
    push!(basis, gamma5*gamma3)
    push!(basis, gamma5*gamma4)



    return basis
end

function Clifford(W::AbstractSpinMatrix)
    basis = Clifford()
    temp = zeros(ComplexF64, 4, 4)
    for i in 1:4, u in 1:4
        temp .+= W[i,u] .* basis[4*(i-1)+u].sm
    end

    return typeof(W)(temp)
end

function Clifford(W::Vector{T}) where{T<:AbstractSpinMatrix}
    
    weights = Vector{T}(undef, 0)

    for i in 1:length(W)
        push!(weights,Clifford(W[i]))
    end


    return weights 


end

# GPU spin matrices, stored as a static 4x4 matrix.

mutable struct Spin_matrix_GPU <: AbstractSpinMatrix
    sm::MMatrix{4,4,ComplexF64}
end

function Random_SpinM_GPU(matrix_type::String = "complex")
    if matrix_type == "complex"
        mat = randn(ComplexF64, 4, 4)
    elseif matrix_type == "real"
        mat = ComplexF64.(randn(4, 4))
    elseif matrix_type == "zero"
        mat = zeros(ComplexF64, 4, 4)
    elseif matrix_type == "Init"
        mat = Matrix{ComplexF64}(I, 4, 4)
    else
        error("unsupported matrix_type; use 'complex', 'real', 'zero' or 'Init'.")
    end

    return Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(mat))
end

function Spin_matrix_GPU(S::Spin_matrix)
    return Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(S.sm))
end

Base.size(S::Spin_matrix_GPU) = size(S.sm)

Base.getindex(S::Spin_matrix_GPU, i::Int, j::Int) = S.sm[i, j]

Base.setindex!(S::Spin_matrix_GPU, v, i::Int, j::Int) =
    (S.sm = setindex(S.sm, v, i, j); S)

@inline function Base.:*(c::Number, a::Spin_matrix_GPU)
    sm = MMatrix{4,4,ComplexF64}(undef)
    @inbounds for j in 1:4, i in 1:4
        sm[i,j] = c * a.sm[i,j]
    end
    return Spin_matrix_GPU(sm)
end

@inline Base.:*(a::Spin_matrix_GPU, c::Number) = c * a

function Base.:*(W::Spin_matrix_GPU, K::Spin_matrix_GPU)
    return Spin_matrix_GPU(W.sm * K.sm)
end

function Base.:*(W::Vector{Spin_matrix_GPU}, K::Vector{Spin_matrix_GPU})
    if length(W) != length(K)
        throw(ArgumentError("W and K must have the same length."))
    end

    result = similar(K)

    for i in 1:length(W)
        result[i] = W[i] * K[i]
    end

    return result
end

# A zero Spin_matrix_GPU, without allocating a Matrix.
@inline function zero_spinmat_gpu()
    sm = MMatrix{4,4,ComplexF64}(undef)
    fill!(sm, zero(ComplexF64))
    return Spin_matrix_GPU(sm)
end

# similar returns zero matrices.
Base.similar(::Spin_matrix_GPU, ::Type{ComplexF64}, dims::Dims{2}) =
    dims == (4,4) ? zero_spinmat_gpu() :
    throw(ArgumentError("Spin_matrix_GPU is always 4x4; got dims=$dims"))

Base.similar(::Spin_matrix_GPU) = zero_spinmat_gpu()

Base.similar(A::Vector{Spin_matrix_GPU}) = [zero_spinmat_gpu() for _ in eachindex(A)]

function Base.:/(W::Spin_matrix_GPU, a::Number)
    Spin_matrix_GPU(W.sm / a)
end

function kernel_spinmat_mul!(
    out::CuDeviceArray{ComplexF64,4},
    M::SMatrix{4,4,ComplexF64,16},
    inp::CuDeviceArray{ComplexF64,4},
    b::Int, r::Int
)
    @inbounds for c = 1:3
        x1 = inp[c, 1, b, r]
        x2 = inp[c, 2, b, r]
        x3 = inp[c, 3, b, r]
        x4 = inp[c, 4, b, r]

        out[c, 1, b, r] = M[1,1]*x1 + M[1,2]*x2 + M[1,3]*x3 + M[1,4]*x4
        out[c, 2, b, r] = M[2,1]*x1 + M[2,2]*x2 + M[2,3]*x3 + M[2,4]*x4
        out[c, 3, b, r] = M[3,1]*x1 + M[3,2]*x2 + M[3,3]*x3 + M[3,4]*x4
        out[c, 4, b, r] = M[4,1]*x1 + M[4,2]*x2 + M[4,3]*x3 + M[4,4]*x4
    end
    return
end

function cudakernel_spinmat_mul!(out, M, inp)
    b = Int(threadIdx().x)
    r = Int(blockIdx().x)
    kernel_spinmat_mul!(out, M, inp, b, r)
end

function Base.:+(a::Spin_matrix_GPU, b::Spin_matrix_GPU)
    return Spin_matrix_GPU(a.sm .+ b.sm)
end

function Base.:+(a::Vector{Spin_matrix_GPU}, b::Vector{Spin_matrix_GPU})
    @assert length(a) == length(b) "a and b must have the same length."

    result = Vector{Spin_matrix_GPU}(undef, length(a))

    @inbounds for i in 1:length(a)
        sm = MMatrix{4,4,ComplexF64}(undef)
        for j in 1:4
            @simd for k in 1:4
                sm[j, k] = a[i].sm[j, k] + b[i].sm[j, k]
            end
        end
        result[i] = Spin_matrix_GPU(sm)
    end

    return result
end

function Base.:-(a::Spin_matrix_GPU,b::Spin_matrix_GPU)

    sm = deepcopy(b.sm)

    sm -= a.sm

    return Spin_matrix_GPU(sm)
end

function Base.:-(a::Vector{Spin_matrix_GPU}, b::Vector{Spin_matrix_GPU})

    if length(a) != length(b)
        throw(ArgumentError("a and b must have the same length."))
    end

    result = similar(a)

    for i in 1:length(a)
        result[i] = a[i] - b[i]
    end

    return result
end

Base.adjoint(S::Spin_matrix_GPU) = Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(adjoint(S.sm)))

Base.adjoint(S::Vector{Spin_matrix_GPU}) = [adjoint(s) for s in S]

function Convertsm(W::Spin_matrix_GPU)
    return Spin_matrix(Matrix{ComplexF64}(W.sm))
end 

function Convertsm(W::Vector{Spin_matrix_GPU})
    return [Convertsm(w) for w in W]
end 

# ============================================================
#  Copying a vector of spin-matrix weights to and from a 4 x 4 x NW CuArray
# ============================================================



"""CPU Vector{Spin_matrix_GPU} → GPU CuArray{ComplexF64, 3} (4,4,NW)"""
function sync_W_to_gpu!(W_gpu::CuArray{ComplexF64, 3}, W::AbstractVector{<:AbstractSpinMatrix})
    n = length(W)
    W_cpu = Array{ComplexF64}(undef, 4, 4, n)
    @inbounds for k in 1:n
        sm = W[k].sm
        for j in 1:4, i in 1:4
            W_cpu[i, j, k] = sm[i, j]
        end
    end
    copyto!(W_gpu, W_cpu)
end

"""GPU CuArray{ComplexF64, 3} → CPU Vector{Spin_matrix_GPU}"""
function sync_W_from_gpu!(W::AbstractVector{<:AbstractSpinMatrix}, W_gpu::CuArray{ComplexF64, 3})
    W_cpu = Array(W_gpu)
    @inbounds for k in eachindex(W)
        sm = W[k].sm
        for j in 1:4, i in 1:4
            sm[i, j] = W_cpu[i, j, k]
        end
    end
end

function sync_W_from_gpu!(W::AbstractVector{<:AbstractSpinMatrix}, W_gpu::CuArray{ComplexF64, 3}, W_cpu_buf::Array{ComplexF64, 3})
    copyto!(W_cpu_buf, W_gpu)
    @inbounds for k in eachindex(W)
        sm = W[k].sm
        for j in 1:4, i in 1:4
            sm[i, j] = W_cpu_buf[i, j, k]
        end
    end
end

function xavier_init!(W::Vector{T}, fan_in::Int, fan_out::Int) where {T<:AbstractArray{ComplexF64,2}}
    scale = sqrt(6 / (fan_in + fan_out))
    for M in W
        for i in 1:size(M, 1), j in 1:size(M, 2)
            real_part = rand() * 2 * scale - scale
            imag_part = rand() * 2 * scale - scale
            M[i, j] = complex(real_part, imag_part)
        end
    end
end
