# 4x4 complex matrices acting on the spinor index (Spin_matrix on the CPU,
# Spin_matrix_GPU on the GPU), their arithmetic, the Clifford basis, weight
# initialisation, and packing of weight vectors to and from the GPU.

mutable struct Spin_matrix <: AbstractSpinMatrix
    sm::Matrix{ComplexF64}
end

# 배열 크기 반환
size(S::Spin_matrix) = size(S.sm)  

# 배열 원소 개수 반환
length(S::Spin_matrix) = length(S.sm)  

# 차원 수 반환 (2차원 행렬이므로 2)
ndims(S::Spin_matrix) = 2

# Overload Base.adjoint for Spin_matrix
Base.adjoint(S::Spin_matrix) = Spin_matrix(adjoint(S.sm))

broadcastable(S::Spin_matrix) = S.sm  # 내부 행렬을 브로드캐스팅 가능하게 변환

function Base.adjoint(S::Vector{Spin_matrix})
    return [adjoint(s) for s in S]
end

function Random_SpinM(matrix_type::String = "complex")
    if matrix_type == "complex"
        # 4x4 복소 랜덤 행렬 생성
        mat = randn(4,4) .+ im*randn(4,4)
    elseif matrix_type == "real"
        # 4x4 실수 랜덤 행렬 생성
        mat = randn(4,4)
        # 모든성분이 0인 행렬 생성 
    elseif matrix_type=="zero"
        mat = zeros(ComplexF64, 4,4)

    elseif matrix_type=="Init"
        mat = Matrix{ComplexF64}(I, 4, 4)
    
    else
        error("지원하지 않는 matrix_type 입니다. 'complex' 혹은 'real'을 사용하세요.")
    end
    
    return Spin_matrix(mat)
end

# getindex 정의
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

    # 1) 첫번째 기저: 단위행렬(Identity)
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

    # 3) gamma2  (복소 항)
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

    # 5) gamma4 (혹은 gamma0)
    gamma4 = Random_SpinM("zero")
    gamma4[1,2] = 1
    gamma4[2,1] = 1
    gamma4[3,4] = -1
    gamma4[4,3] = -1
    push!(basis, gamma4)

    # 6) gamma5 (단순 예: 대각에 -1, -1, +1, +1 인 경우)
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

    # 8) gamma^5 gamma^mu (4개)
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

    return typeof(W)(temp)  # 원래 타입으로 래핑
end

function Clifford(W::Vector{T}) where{T<:AbstractSpinMatrix}
    
    weights = Vector{T}(undef, 0)

    for i in 1:length(W)
        push!(weights,Clifford(W[i]))
    end


    return weights 


end

# GPU 버전

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
        error("지원하지 않는 matrix_type 입니다. 'complex', 'real', 'zero', 'Init' 중 선택하세요.")
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
    # W는 Spin_matrix_GPU, K는 Spin_matrix_GPU
    return Spin_matrix_GPU(W.sm * K.sm)
end

function Base.:*(W::Vector{Spin_matrix_GPU}, K::Vector{Spin_matrix_GPU})
    if length(W) != length(K)
        throw(ArgumentError("W와 K의 길이가 일치하지 않습니다."))
    end

    result = similar(K)

    for i in 1:length(W)
        result[i] = W[i] * K[i]
    end

    return result
end

# 항상 0으로 초기화된 Spin_matrix_GPU 생성 (Matrix 할당 없이)
@inline function zero_spinmat_gpu()
    sm = MMatrix{4,4,ComplexF64}(undef)
    fill!(sm, zero(ComplexF64))
    return Spin_matrix_GPU(sm)
end

# (Spin_matrix_GPU, ComplexF64, (4,4)) -> 0행렬
Base.similar(::Spin_matrix_GPU, ::Type{ComplexF64}, dims::Dims{2}) =
    dims == (4,4) ? zero_spinmat_gpu() :
    throw(ArgumentError("Spin_matrix_GPU는 4×4만 지원합니다. dims=$dims"))

# Spin_matrix_GPU 하나의 similar도 필요하면 이것도 추가 (0행렬)
Base.similar(::Spin_matrix_GPU) = zero_spinmat_gpu()

# Vector{Spin_matrix_GPU}의 similar도 각 원소를 0행렬로
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
    return Spin_matrix_GPU(a.sm .+ b.sm)  # ⬅ broadcast 연산으로 StaticArrays 최적화 사용
end

function Base.:+(a::Vector{Spin_matrix_GPU}, b::Vector{Spin_matrix_GPU})
    @assert length(a) == length(b) "a와 b의 길이가 일치하지 않습니다."

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
        throw(ArgumentError("a와 b의 길이가 일치하지 않습니다."))
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
#  Batched PTLayer kernels — GPU-resident W, 단일 커널 launch
# ============================================================


# ---- W ↔ CuArray 동기화 ----

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
