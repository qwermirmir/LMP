module gpu




# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using CUDA
using StaticArrays

import Gaugefields:add_U!,clear_U!

import ..elementry_base: Spin_matrix, Hopping_operator, Random_SpinM,AbstractSpinMatrix


const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}
const GPUgauge=Vector{Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, :cuda, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}}


const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

using StaticArrays

export Spin_matrix_GPU, Random_SpinM_GPU,Generate_weight
export Matmul!,Convertsm,gpuaddmul!,conj!,zero_wf!



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








function Base.:*(W::Vector{Spin_matrix_GPU}, x::Vector{WF3x1GPU})
    # W는 Spin_matrix_GPU의 벡터, x는 WF3x1GPU의 벡터
    if length(W) != length(x)
        throw(ArgumentError("W와 x의 길이가 일치하지 않습니다."))
    end

    result = similar(x)

    for i in 1:length(W)
        result[i] = W[i] * x[i]
    end

    return result
end
function Base.:*(c::Number,a::WF3x1GPU)


    b=similar(a)

    mul!(b,c,a)
    
    return b
end 



function Base.:*(W::Vector{Spin_matrix_GPU}, x::Vector{Union{WF3x1GPU, WF3x1CPU}})
    if length(W) != length(x)
        throw(ArgumentError("W와 x의 길이가 일치하지 않습니다."))
    end

    result = similar(x)

    for i in 1:length(W)
        if x[i] isa WF3x1GPU
            result[i] = W[i] * x[i]  # GPU 연산
        else
            throw(ArgumentError("알 수 없는 타입: $(typeof(x[i]))"))
        end
    end

    return result
end

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





function norm2(x::WF3x1GPU)
    # x는 WF3x1GPU 타입
    return dot(x, x)
end

function Base.conj(x::WF3x1GPU)
    out = deepcopy(x)
    total = length(x.f)
    threads = 256
    blocks  = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks kernel_conj_1D!(out.f, x.f)
    return out
end




# CUDA 커널 정의
function kernel_conj_1D!(
    out::CuDeviceArray{ComplexF64, 4},
    inp::CuDeviceArray{ComplexF64, 4}
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(inp)

    if idx <= total
        out[idx] = conj(inp[idx])
    end

    return nothing
end

function conj!(out::WF3x1GPU, inp::WF3x1GPU)
    total = length(inp.f)
    threads = 256
    blocks  = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks kernel_conj_1D!(out.f, inp.f)
    return out
end


function conj!(x::WF3x1CPU,y::WF3x1CPU)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    NG = x.NG

    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            x[ic, ix, iy, iz, it, ig] = conj(y[ic, ix, iy, iz, it, ig])
                        end
                    end
                end
            end
        end
    end

end








function Generate_weight(x::Any; type="real")
    W = Vector{Spin_matrix}()

    if x isa Vector{WF3x1}
        for _ in x
            push!(W, Random_SpinM(type))
        end

    elseif x isa WF3x1
        push!(W, Random_SpinM(type))

    elseif x isa Hopping_operator
        U = x.Gauge
        if U isa GPUgauge
            W = [Random_SpinM_GPU(type) for _ in 1:length(x.P.various)]
        else
            W = [Random_SpinM(type) for _ in 1:length(x.P.various)]
        end

    else
        throw(ArgumentError("지원되지 않는 x 타입: $(typeof(x))"))
    end

    return W
end





function (hop::Hopping_operator)(x::WF3x1GPU)

    shifts = Vector{WF3x1GPU}()

    for k in 1:length(hop.P.various)
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                mul!(copyx,hop.Gauge'[a],copyx)
                shift_x=shift_fermion(copyx , -a)
                substitute_fermion!(copyx,shift_x)

            elseif a < 0
                a = -a
                shift_x=shift_fermion(copyx, a)
                mul!(copyx, hop.Gauge[a], shift_x)
            elseif a == 0
                copyx = deepcopy(x)     
            end
        end
        push!(shifts,copyx)
    end

    return shifts
end

function (hop::Hopping_operator)( x::WF3x1GPU,V::AbstractVector{T}) where {T<:WF3x1GPU}
    @assert length(V) == length(hop.P.various)

    @inbounds for k in eachindex(V)
        copyx = V[k]

        # copyx를 "x로 초기화" (새로 할당 X)
        substitute_fermion!(copyx, x)   # 당신 코드에 이미 있는 in-place copy라고 가정

        for step in hop.P.various[k].steps
            a = step.f

            if a > 0
                mul!(copyx, adjoint(hop.Gauge[a]), copyx)

                # 여기 shift_fermion이 out-of-place면 여기가 또 할당 포인트
                shift_x = shift_fermion(copyx, -a)
                substitute_fermion!(copyx, shift_x)

            elseif a < 0
                aa = -a
                shift_x = shift_fermion(copyx, aa)
                mul!(copyx, hop.Gauge[aa], shift_x)

            else
                substitute_fermion!(copyx, x)
            end
        end
    end

    return V
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



function Base.:*(M::Spin_matrix_GPU, x::WF3x1GPU)
    f  = deepcopy(x)
    Ms = SMatrix{4,4,ComplexF64}(M.sm)   # 16개 값, 커널 인자로 “값” 전달 (CuArray 생성 X)

    CUDA.@cuda threads=x.blockinfo.blocksize blocks=x.blockinfo.rsize cudakernel_spinmat_mul!(f.f, Ms, x.f)

    return f
end




function Base.:+(a::WF3x1GPU,b::WF3x1GPU)

    f =deepcopy(b)



    axpby!(1,a,1,f)

    return f
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



function Base.:-(a::WF3x1GPU,b::WF3x1GPU)

    y=deepcopy(b)



    axpby!(1,a,-1,y)

    return y
end 
function Base.conj(W::Spin_matrix_GPU)
    return Spin_matrix_GPU(conj(W.sm))
end


# ====== Batched Matmul: N 개의 (Q,K) 쌍을 한번에 계산, GPU→CPU sync 1회 ======
function kernel_matmul_batched!(
    M_all::CuDeviceArray{ComplexF64,3},   # (NG, NG, batch)
    Q_ptrs::CuDeviceArray{ComplexF64,4},  # 현재 Q의 f
    K_ptrs::CuDeviceArray{ComplexF64,4},  # 현재 K의 f
    NC::Int, NG::Int, blocksize::Int, rsize::Int,
    batch_idx::Int
)
    tid = threadIdx().x
    bid = blockIdx().x

    fg = (bid - 1) % NG + 1
    xg = (bid - 1) ÷ NG + 1
    if xg > NG
        return
    end

    N = NC * blocksize * rsize
    acc = zero(ComplexF64)

    stride = blockDim().x
    k = tid
    while k <= N
        k0 = k - 1
        ic = (k0 % NC) + 1
        t  = k0 ÷ NC
        b  = (t % blocksize) + 1
        r  = (t ÷ blocksize) + 1

        @inbounds acc += Q_ptrs[ic, fg, b, r] * K_ptrs[ic, xg, b, r]
        k += stride
    end

    sh = CUDA.CuStaticSharedArray(ComplexF64, 256)
    @inbounds sh[tid] = acc
    CUDA.sync_threads()

    offset = stride >>> 1
    while offset >= 1
        if tid <= offset
            @inbounds sh[tid] += sh[tid + offset]
        end
        CUDA.sync_threads()
        offset >>>= 1
    end

    if tid == 1
        @inbounds M_all[fg, xg, batch_idx] = sh[1]
    end
    return nothing
end

"""
    Matmul_batch(Qs, Ks) → Vector{Spin_matrix_GPU}

N개의 (Q,K) 쌍에 대해 Matmul을 계산. GPU→CPU sync는 1회만 수행.
"""
function Matmul_batch(Qs::AbstractVector{<:WF3x1GPU}, Ks::AbstractVector{<:WF3x1GPU})
    n = length(Qs)
    @assert n == length(Ks)
    NC, NG = Qs[1].NC, Qs[1].NG
    blocksize = Qs[1].blockinfo.blocksize
    rsize     = Qs[1].blockinfo.rsize

    M_all = CUDA.zeros(ComplexF64, NG, NG, n)

    threads = 256
    blocks  = NG * NG

    for idx in 1:n
        CUDA.@cuda threads=threads blocks=blocks kernel_matmul_batched!(
            M_all, Qs[idx].f, Ks[idx].f, NC, NG, blocksize, rsize, idx)
    end

    # GPU→CPU sync 1회
    M_cpu = Array(M_all)

    result = Vector{Spin_matrix_GPU}(undef, n)
    @inbounds for idx in 1:n
        result[idx] = Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(@view M_cpu[:,:,idx]))
    end
    return result
end

"""
    Matmul_batch!(result, M_all, M_cpu, Qs, Ks) → result

Preallocated version of Matmul_batch. Reuses M_all (CuArray), M_cpu (CPU Array), result (Vector{Spin_matrix_GPU}).
"""
function Matmul_batch!(result, M_all, M_cpu::Array{ComplexF64,3},
                       Qs::AbstractVector, Ks::AbstractVector)
    n = length(Qs)
    NC, NG = Qs[1].NC, Qs[1].NG
    blocksize = Qs[1].blockinfo.blocksize
    rsize     = Qs[1].blockinfo.rsize

    M_all .= 0

    threads = 256
    blocks  = NG * NG

    for idx in 1:n
        CUDA.@cuda threads=threads blocks=blocks kernel_matmul_batched!(
            M_all, Qs[idx].f, Ks[idx].f, NC, NG, blocksize, rsize, idx)
    end

    copyto!(M_cpu, M_all)

    @inbounds for idx in 1:n
        result[idx].sm .= MMatrix{4,4,ComplexF64}(@view M_cpu[:,:,idx])
    end
    return result
end

export Matmul_batch, Matmul_batch!







function kernel_matmul!(
    M::CuDeviceMatrix{ComplexF64},
    Q::CuDeviceArray{ComplexF64,4},   # (ic, mu, b, r)
    K::CuDeviceArray{ComplexF64,4},   # (ic, mu, b, r)
    NC::Int, NG::Int, blocksize::Int, rsize::Int
)
    tid = threadIdx().x
    bid = blockIdx().x

    # block id -> (fg, xg)  (1-based)
    fg = (bid - 1) % NG + 1
    xg = (bid - 1) ÷ NG + 1
    if xg > NG
        return
    end

    N = NC * blocksize * rsize

    # 각 스레드의 부분합
    acc = zero(ComplexF64)

    # flatten index k -> (ic,b,r)
    stride = blockDim().x
    k = tid
    while k <= N
        k0 = k - 1
        ic = (k0 % NC) + 1
        t  = k0 ÷ NC
        b  = (t % blocksize) + 1
        r  = (t ÷ blocksize) + 1

        @inbounds acc += Q[ic, fg, b, r] * K[ic, xg, b, r]

        k += stride
    end

    # 블록 내 reduction (shared memory)
    sh = CUDA.CuStaticSharedArray(ComplexF64, 256)  # threads=256 가정
    @inbounds sh[tid] = acc
    CUDA.sync_threads()

    offset = stride >>> 1
    while offset >= 1
        if tid <= offset
            @inbounds sh[tid] += sh[tid + offset]
        end
        CUDA.sync_threads()
        offset >>>= 1
    end

    if tid == 1
        @inbounds M[fg, xg] = sh[1]
    end
    return nothing
end



function Matmul!(Q::WF3x1GPU, K::WF3x1GPU)
    NC, NG = Q.NC, Q.NG
    blocksize = Q.blockinfo.blocksize
    rsize     = Q.blockinfo.rsize

    M_d = CUDA.zeros(ComplexF64, NG, NG)

    threads = 256
    blocks  = NG * NG

    CUDA.@cuda threads=threads blocks=blocks kernel_matmul!(M_d, Q.f, K.f, NC, NG, blocksize, rsize)

    # 4×4만 가져오니 전송 자체는 작음(하지만 호출이 너무 잦으면 여기서도 손해)
    M_cpu = Array(M_d)
    return Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(M_cpu))
end


function Matmul!(Q::WF3x1CPU,K::WF3x1CPU)

    M=Random_SpinM("zero")

    

    NX = Q.NX
    NY = Q.NY
    NZ = Q.NZ
    NT = Q.NT
    NC = Q.NC
    NG = Q.NG

    


    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for ic = 1:NC
                        # Q와 K의 각 성분을 곱하여 M에 누적
                        for fg = 1:NG
                            for xg= 1:NG
                                M[fg,xg]+=Q[ic,ix,iy,iz,it,fg]*K[ic,ix,iy,iz,it,xg]
                            end
                        end
                    end
                end
            end
        end
    end


    return M
    
end 



Base.adjoint(S::Spin_matrix_GPU) = Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(adjoint(S.sm)))

Base.adjoint(S::Vector{Spin_matrix_GPU}) = [adjoint(s) for s in S]

function Convertsm(W::Spin_matrix_GPU)
    return Spin_matrix(Matrix{ComplexF64}(W.sm))
end 

function Convertsm(W::Vector{Spin_matrix_GPU})
    return [Convertsm(w) for w in W]
end 


export substitute_fermion!
import LatticeDiracOperators.Dirac_operators: substitute_fermion!, Shifted_fermionfields_4D_accelerator

# GPU: dest(::WF3x1GPU) <- src.parent.fshifted  (device-to-device copy)
function substitute_fermion!(
    dest::WF3x1GPU,
    src::Shifted_fermionfields_4D_accelerator{NC,<:WF3x1GPU}
) where {NC}
    parent = src.parent
    @assert parent.fshifted !== nothing "noshiftfield=true라서 fshifted 버퍼가 없습니다."
    copyto!(dest.f, parent.fshifted)   # ✅ GPU memcpy
    return dest
end


# WF3x1GPU 전용 addmul! 커널
function kernel_addmul_spinmat_mul!(
    out::CuDeviceArray{ComplexF64,4},   # (c, α, b, r)
    M::SMatrix{4,4,ComplexF64,16},
    inp::CuDeviceArray{ComplexF64,4},
)
    b = Int(threadIdx().x)
    r = Int(blockIdx().x)

    @inbounds for c = 1:3
        x1 = inp[c, 1, b, r]
        x2 = inp[c, 2, b, r]
        x3 = inp[c, 3, b, r]
        x4 = inp[c, 4, b, r]

        out[c, 1, b, r] += M[1,1]*x1 + M[1,2]*x2 + M[1,3]*x3 + M[1,4]*x4
        out[c, 2, b, r] += M[2,1]*x1 + M[2,2]*x2 + M[2,3]*x3 + M[2,4]*x4
        out[c, 3, b, r] += M[3,1]*x1 + M[3,2]*x2 + M[3,3]*x3 + M[3,4]*x4
        out[c, 4, b, r] += M[4,1]*x1 + M[4,2]*x2 + M[4,3]*x3 + M[4,4]*x4
    end
    return nothing
end

# WF3x1GPU 전용 addmul!
function gpuaddmul!(out::WF3x1GPU, M::Spin_matrix_GPU, inp::WF3x1GPU)
    Ms = SMatrix{4,4,ComplexF64}(M.sm)  # isbits, 커널 인자로 OK
    CUDA.@cuda threads=out.blockinfo.blocksize blocks=out.blockinfo.rsize kernel_addmul_spinmat_mul!(out.f, Ms, inp.f)
    return out
end

function zero_wf!(x::WF3x1GPU)
    total = length(x.f)
    threads = 256
    blocks  = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks kernel_zero!(x.f)
end 

function kernel_zero!(
    out::CuDeviceArray{ComplexF64, 4},

)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(out)

    if idx <= total
        out[idx] = 0
    end

    return nothing
end

function zero_wf!(v::Vector{<:WF3x1GPU})
    @inbounds for i in eachindex(v)
        zero_wf!(v[i])
    end
end 


function zero_wf!(x::WF3x1CPU)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    NG = x.NG

    z0 = zero(typeof(x[1,1,1,1,1,1]))  # 예: 0.0 + 0.0im

    @inbounds for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            x[ic, ix, iy, iz, it, ig] = z0
                        end
                    end
                end
            end
        end
    end

    return x
end

function zero_wf!(v::AbstractVector{<:WF3x1})
    @inbounds for i in eachindex(v)
        zero_wf!(v[i])
    end
    return v
end


"""
    _build_block_weight_gpu(W_col, r, x::WF3x1GPU)

BlockScalar_Layer용: GPU 페르미온 `.f`(shape NC×NG×blocksize×rsize)에
broadcast 곱셈 가능한 가중치 CuArray(shape 1×1×blocksize×rsize)를 생성.

`W_col[i]`는 블록 i의 가중치, `r`은 blockize 블록 크기.
Blockindices의 (b,r)→(ix,iy,iz,it) 매핑을 사용.
"""
function _build_block_weight_gpu(W_col::AbstractVector{<:Number}, r::Int, x::WF3x1GPU)
    bi = x.blockinfo
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r

    blocksize = bi.blocksize
    rsize     = bi.rsize

    w_cpu = zeros(ComplexF64, blocksize, rsize)

    @inbounds for r_idx = 1:rsize
        for b_idx = 1:blocksize
            ix = 1 + mod(div(b_idx-1, bi.blocks_s[1]), bi.blocks[1]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[1]), bi.blocknumbers[1]) * bi.blocks[1]
            iy = 1 + mod(div(b_idx-1, bi.blocks_s[2]), bi.blocks[2]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[2]), bi.blocknumbers[2]) * bi.blocks[2]
            iz = 1 + mod(div(b_idx-1, bi.blocks_s[3]), bi.blocks[3]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[3]), bi.blocknumbers[3]) * bi.blocks[3]
            it = 1 + mod(div(b_idx-1, bi.blocks_s[4]), bi.blocks[4]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[4]), bi.blocknumbers[4]) * bi.blocks[4]

            bx = (ix - 1) ÷ r + 1
            by = (iy - 1) ÷ r + 1
            bz = (iz - 1) ÷ r + 1
            bt = (it - 1) ÷ r + 1
            block_id = bx + NBX * ((by - 1) + NBY * ((bz - 1) + NBZ * (bt - 1)))

            w_cpu[b_idx, r_idx] = W_col[block_id]
        end
    end

    return CuArray(reshape(w_cpu, 1, 1, blocksize, rsize))
end

"""
    _init_gpu_block_cache!(layer, x)

GPU용 block 인덱스 맵 + 버퍼를 1회만 초기화.
`layer._gpu_block_map[linear_idx]` = block_id 로, W 값만 갱신하면 됨.
"""
function _init_gpu_block_cache!(layer, x::WF3x1GPU)
    bi = x.blockinfo
    r = layer.r
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NBX = NX÷r; NBY = NY÷r; NBZ = NZ÷r; NBT = NT÷r
    blocksize = bi.blocksize
    rsize = bi.rsize

    bmap = Vector{Int}(undef, blocksize * rsize)
    idx = 0
    @inbounds for r_idx = 1:rsize
        for b_idx = 1:blocksize
            idx += 1
            ix = 1 + mod(div(b_idx-1, bi.blocks_s[1]), bi.blocks[1]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[1]), bi.blocknumbers[1]) * bi.blocks[1]
            iy = 1 + mod(div(b_idx-1, bi.blocks_s[2]), bi.blocks[2]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[2]), bi.blocknumbers[2]) * bi.blocks[2]
            iz = 1 + mod(div(b_idx-1, bi.blocks_s[3]), bi.blocks[3]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[3]), bi.blocknumbers[3]) * bi.blocks[3]
            it = 1 + mod(div(b_idx-1, bi.blocks_s[4]), bi.blocks[4]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[4]), bi.blocknumbers[4]) * bi.blocks[4]
            bx = (ix-1)÷r + 1; by = (iy-1)÷r + 1
            bz = (iz-1)÷r + 1; bt = (it-1)÷r + 1
            bmap[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
        end
    end
    layer._gpu_block_map = bmap
    layer._gpu_w_cpu = zeros(ComplexF64, blocksize, rsize)
    layer._gpu_w_buf = CUDA.zeros(ComplexF64, 1, 1, blocksize, rsize)
end

"""
    _update_block_weight_gpu!(layer, W_col)

캐시된 맵으로 CPU 배열 갱신 → CuArray에 copyto! (할당 0)
"""
function _update_block_weight_gpu!(layer, W_col::AbstractVector{<:Number})
    bmap = layer._gpu_block_map
    w_cpu = layer._gpu_w_cpu
    @inbounds for i in eachindex(bmap)
        w_cpu[i] = W_col[bmap[i]]
    end
    copyto!(layer._gpu_w_buf, reshape(w_cpu, 1, 1, size(w_cpu)...))
    return layer._gpu_w_buf
end

export _build_block_weight_gpu, _init_gpu_block_cache!, _update_block_weight_gpu!


# ============================================================
#  Batched PTLayer kernels — GPU-resident W, 단일 커널 launch
# ============================================================

export sync_W_to_gpu!, sync_W_from_gpu!
export pack_fermions!, unpack_fermions!
export batched_forward!, batched_gradient!, batched_delta!
export gpu_adam_update!

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

# ---- Pack / Unpack fermion fields ----

"""Vector{WF3x1GPU} → CuArray{ComplexF64, 5} (NC,NG,bs,rs,n)"""
function pack_fermions!(buf::CuArray{ComplexF64, 5}, fields::AbstractVector)
    @inbounds for i in eachindex(fields)
        copyto!(@view(buf[:,:,:,:,i]), fields[i].f)
    end
end

"""CuArray{ComplexF64, 5} → Vector{WF3x1GPU}  (각 field의 .f에 복사)"""
function unpack_fermions!(fields::AbstractVector, buf::CuArray{ComplexF64, 5})
    @inbounds for j in eachindex(fields)
        copyto!(fields[j].f, @view(buf[:,:,:,:,j]))
    end
end


# ──── Batched forward kernel ────
# result[c,g,b,r,j] = Σ_i Σ_g2  W[g,g2,(i-1)*Out+j] * hopped[c,g2,b,r,i]

function _cudakernel_batched_forward!(
    result_all::CuDeviceArray{ComplexF64, 5},
    W_gpu     ::CuDeviceArray{ComplexF64, 3},
    hopped_all::CuDeviceArray{ComplexF64, 5},
    NC::Int, NG::Int, npaths::Int, Out::Int
)
    b = Int(threadIdx().x)
    r = Int(blockIdx().x)

    @inbounds for j in 1:Out
        for c in 1:NC
            a1 = zero(ComplexF64); a2 = zero(ComplexF64)
            a3 = zero(ComplexF64); a4 = zero(ComplexF64)
            for i in 1:npaths
                widx = (i-1)*Out + j
                x1 = hopped_all[c, 1, b, r, i]
                x2 = hopped_all[c, 2, b, r, i]
                x3 = hopped_all[c, 3, b, r, i]
                x4 = hopped_all[c, 4, b, r, i]
                a1 += W_gpu[1,1,widx]*x1 + W_gpu[1,2,widx]*x2 + W_gpu[1,3,widx]*x3 + W_gpu[1,4,widx]*x4
                a2 += W_gpu[2,1,widx]*x1 + W_gpu[2,2,widx]*x2 + W_gpu[2,3,widx]*x3 + W_gpu[2,4,widx]*x4
                a3 += W_gpu[3,1,widx]*x1 + W_gpu[3,2,widx]*x2 + W_gpu[3,3,widx]*x3 + W_gpu[3,4,widx]*x4
                a4 += W_gpu[4,1,widx]*x1 + W_gpu[4,2,widx]*x2 + W_gpu[4,3,widx]*x3 + W_gpu[4,4,widx]*x4
            end
            result_all[c, 1, b, r, j] = a1
            result_all[c, 2, b, r, j] = a2
            result_all[c, 3, b, r, j] = a3
            result_all[c, 4, b, r, j] = a4
        end
    end
    return nothing
end

"""
    batched_forward!(result_all, W_gpu, hopped_all, blocksize, rsize, npaths, Out)

625개 gpuaddmul! 대신 1회 커널 launch.
"""
function batched_forward!(result_all, W_gpu, hopped_all, blocksize, rsize, npaths, Out)
    NC = 3; NG = 4
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_forward!(
        result_all, W_gpu, hopped_all, NC, NG, npaths, Out)
end


# ──── Batched gradient kernel (inner-product reduction) ────
# M_all[fg,xg,widx] = Σ_{c,b,r} d[c,fg,b,r,j] * hopped[c,xg,b,r,i]
#  where  j = (widx-1)%Out + 1,  i = (widx-1)÷Out + 1

function _cudakernel_batched_gradient!(
    M_all     ::CuDeviceArray{ComplexF64, 3},
    d_all     ::CuDeviceArray{ComplexF64, 5},
    hopped_all::CuDeviceArray{ComplexF64, 5},
    NC::Int, NG::Int, blocksize::Int, rsize::Int, Out::Int, npaths::Int
)
    tid = Int(threadIdx().x)
    bid = Int(blockIdx().x)

    NW = npaths * Out
    fg  = ((bid - 1) % NG) + 1
    rest = (bid - 1) ÷ NG
    xg   = (rest % NG) + 1
    widx = (rest ÷ NG) + 1
    if widx > NW; return; end

    j = ((widx - 1) % Out) + 1
    i = ((widx - 1) ÷ Out) + 1

    N = NC * blocksize * rsize
    acc = zero(ComplexF64)
    stride = blockDim().x
    k = tid
    while k <= N
        k0 = k - 1
        ic = (k0 % NC) + 1
        t  = k0 ÷ NC
        b  = (t % blocksize) + 1
        r  = (t ÷ blocksize) + 1
        @inbounds acc += d_all[ic, fg, b, r, j] * hopped_all[ic, xg, b, r, i]
        k += stride
    end

    sh = CUDA.CuStaticSharedArray(ComplexF64, 256)
    @inbounds sh[tid] = acc
    CUDA.sync_threads()
    offset = stride >>> 1
    while offset >= 1
        if tid <= offset
            @inbounds sh[tid] += sh[tid + offset]
        end
        CUDA.sync_threads()
        offset >>>= 1
    end
    if tid == 1
        @inbounds M_all[fg, xg, widx] = sh[1]
    end
    return nothing
end

"""
    batched_gradient!(grad_gpu, d_all, hopped_all, blocksize, rsize, Out, npaths)

625개 kernel_matmul_batched! 대신 1회 launch.
결과는 grad_gpu (CuArray 4×4×NW) 에 직접 기록 → GPU→CPU 전송 불필요.
"""
function batched_gradient!(grad_gpu, d_all, hopped_all, blocksize, rsize, Out, npaths)
    NG = 4; NC = 3
    NW = npaths * Out
    threads = 256
    nblocks = NG * NG * NW          # 16 * NW
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_batched_gradient!(
        grad_gpu, d_all, hopped_all, NC, NG, blocksize, rsize, Out, npaths)
end


# ──── Batched DeltaRule kernel ────
# ci[c,g,b,r,i] = Σ_j Σ_g2  Wadj[g,g2,(i-1)*Out+j] * conj(d[c,g2,b,r,j])

function _cudakernel_batched_delta!(
    ci_all   ::CuDeviceArray{ComplexF64, 5},
    Wadj_gpu ::CuDeviceArray{ComplexF64, 3},
    d_all    ::CuDeviceArray{ComplexF64, 5},
    NC::Int, NG::Int, Out::Int, npaths::Int
)
    b = Int(threadIdx().x)
    r = Int(blockIdx().x)

    @inbounds for i in 1:npaths
        for c in 1:NC
            a1 = zero(ComplexF64); a2 = zero(ComplexF64)
            a3 = zero(ComplexF64); a4 = zero(ComplexF64)
            for j in 1:Out
                widx = (i-1)*Out + j
                x1 = conj(d_all[c, 1, b, r, j])
                x2 = conj(d_all[c, 2, b, r, j])
                x3 = conj(d_all[c, 3, b, r, j])
                x4 = conj(d_all[c, 4, b, r, j])
                a1 += Wadj_gpu[1,1,widx]*x1 + Wadj_gpu[1,2,widx]*x2 + Wadj_gpu[1,3,widx]*x3 + Wadj_gpu[1,4,widx]*x4
                a2 += Wadj_gpu[2,1,widx]*x1 + Wadj_gpu[2,2,widx]*x2 + Wadj_gpu[2,3,widx]*x3 + Wadj_gpu[2,4,widx]*x4
                a3 += Wadj_gpu[3,1,widx]*x1 + Wadj_gpu[3,2,widx]*x2 + Wadj_gpu[3,3,widx]*x3 + Wadj_gpu[3,4,widx]*x4
                a4 += Wadj_gpu[4,1,widx]*x1 + Wadj_gpu[4,2,widx]*x2 + Wadj_gpu[4,3,widx]*x3 + Wadj_gpu[4,4,widx]*x4
            end
            ci_all[c, 1, b, r, i] = a1
            ci_all[c, 2, b, r, i] = a2
            ci_all[c, 3, b, r, i] = a3
            ci_all[c, 4, b, r, i] = a4
        end
    end
    return nothing
end

"""
    batched_delta!(ci_all, Wadj_gpu, d_all, blocksize, rsize, Out, npaths)

DeltaRule의 gpuaddmul! 625회 → 1회 커널.
d_all은 원본 d (conj는 커널 내부에서 수행).
"""
function batched_delta!(ci_all, Wadj_gpu, d_all, blocksize, rsize, Out, npaths)
    NC = 3; NG = 4
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_delta!(
        ci_all, Wadj_gpu, d_all, NC, NG, Out, npaths)
end


# ──── GPU-side Adam update ────

function _cudakernel_adam_complex!(
    W_gpu   ::CuDeviceArray{ComplexF64, 3},
    m_gpu   ::CuDeviceArray{ComplexF64, 3},
    v_gpu   ::CuDeviceArray{ComplexF64, 3},
    grad_gpu::CuDeviceArray{ComplexF64, 3},
    α::Float64, β1::Float64, β2::Float64, ε::Float64,
    bias1::Float64, bias2::Float64
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(W_gpu)
    if idx > total; return; end

    @inbounds begin
        g = grad_gpu[idx]
        g_re = real(g); g_im = imag(g)

        m = m_gpu[idx]
        mr = β1 * real(m) + (1 - β1) * g_re
        mi = β1 * imag(m) + (1 - β1) * g_im

        v = v_gpu[idx]
        vr = β2 * real(v) + (1 - β2) * g_re * g_re
        vi = β2 * imag(v) + (1 - β2) * g_im * g_im

        m_gpu[idx] = complex(mr, mi)
        v_gpu[idx] = complex(vr, vi)

        upd_re = α * (mr * bias1) / (sqrt(vr * bias2) + ε)
        upd_im = α * (mi * bias1) / (sqrt(vi * bias2) + ε)

        W_gpu[idx] -= complex(upd_re, -upd_im)
    end
    return nothing
end

"""
    gpu_adam_update!(W_gpu, m_gpu, v_gpu, grad_gpu, opt)

GPU 위에서 Adam update를 1회 커널로 수행. CPU 전송 0.
"""
function gpu_adam_update!(W_gpu, m_gpu, v_gpu, grad_gpu,
                          α, β1, β2, ε, t)
    bias1 = 1.0 / (1.0 - β1^t)
    bias2 = 1.0 / (1.0 - β2^t)
    total = length(W_gpu)
    threads = min(256, total)
    nblocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adam_complex!(
        W_gpu, m_gpu, v_gpu, grad_gpu, α, β1, β2, ε, bias1, bias2)
end

# ---- Wadj 동기화 커널 ----
function _cudakernel_adjoint_W!(Wadj, W, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n; return; end
    @inbounds for j in 1:4, i in 1:4
        Wadj[i, j, idx] = conj(W[j, i, idx])
    end
    return nothing
end

"""W_gpu → Wadj_gpu (transpose-conjugate, GPU 위에서)"""
function sync_Wadj_gpu!(Wadj_gpu, W_gpu)
    n = size(W_gpu, 3)
    threads = min(256, n)
    nblocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adjoint_W!(Wadj_gpu, W_gpu, n)
end

export sync_Wadj_gpu!


# ============================================================
#  Fused GPU hop kernels — 3 kernels per step → 1
# ============================================================

# Inline helper functions from Gaugefields blockinfo

@inline _cvt_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)

@inline function _si_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b; rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == blocks - 1
        bs = bs - (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == blocknumbers - 1
            rs = rs - (blocknumbers - 1) * blocknumbers_s
        else
            rs += blocknumbers_s
        end
    else
        bs += blocks_s
    end
    return bs, rs
end

@inline function _si_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b; rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == 0
        bs = bs + (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == 0
            rs = rs + (blocknumbers - 1) * blocknumbers_s
        else
            rs -= blocknumbers_s
        end
    else
        bs -= blocks_s
    end
    return bs, rs
end

@inline function _shifted_idx(b, r, shift, blockinfo)
    bs = b; rs = r
    for idim = 1:4
        blk   = blockinfo.blocks[idim]
        blk_s = blockinfo.blocks_s[idim]
        bn    = blockinfo.blocknumbers[idim]
        bn_s  = blockinfo.blocknumbers_s[idim]
        ns = abs(shift[idim])
        if shift[idim] > 0
            for _ = 1:ns
                bs, rs = _si_plus(bs, rs, blk, blk_s, bn, bn_s)
            end
        else
            for _ = 1:ns
                bs, rs = _si_minus(bs, rs, blk, blk_s, bn, bn_s)
            end
        end
    end
    return bs, rs
end

@inline function _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix = 1 + _cvt_x(b, blockinfo.blocks[1], blockinfo.blocks_s[1]) +
             _cvt_x(r, blockinfo.blocknumbers[1], blockinfo.blocknumbers_s[1]) * blockinfo.blocks[1]
    iy = 1 + _cvt_x(b, blockinfo.blocks[2], blockinfo.blocks_s[2]) +
             _cvt_x(r, blockinfo.blocknumbers[2], blockinfo.blocknumbers_s[2]) * blockinfo.blocks[2]
    iz = 1 + _cvt_x(b, blockinfo.blocks[3], blockinfo.blocks_s[3]) +
             _cvt_x(r, blockinfo.blocknumbers[3], blockinfo.blocknumbers_s[3]) * blockinfo.blocks[3]
    it = 1 + _cvt_x(b, blockinfo.blocks[4], blockinfo.blocks_s[4]) +
             _cvt_x(r, blockinfo.blocknumbers[4], blockinfo.blocknumbers_s[4]) * blockinfo.blocks[4]
    f = ifelse(ix + shift[1] > NX || ix + shift[1] < 1, bc[1], Int8(1)) *
        ifelse(iy + shift[2] > NY || iy + shift[2] < 1, bc[2], Int8(1)) *
        ifelse(iz + shift[3] > NZ || iz + shift[3] < 1, bc[3], Int8(1)) *
        ifelse(it + shift[4] > NT || it + shift[4] < 1, bc[4], Int8(1))
    return f
end

# Fused kernel: positive direction (a > 0)
# y(b,r) = U†_a(bs,rs) * x(bs,rs) * factor
function _cudakernel_fused_hop_pos!(y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    bs, rs = _shifted_idx(b, r, shift, blockinfo)
    fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    @inbounds for alpha = 1:4
        x1 = x[1, alpha, bs, rs] * fac
        x2 = x[2, alpha, bs, rs] * fac
        x3 = x[3, alpha, bs, rs] * fac
        y[1, alpha, b, r] = conj(A[1,1,bs,rs])*x1 + conj(A[2,1,bs,rs])*x2 + conj(A[3,1,bs,rs])*x3
        y[2, alpha, b, r] = conj(A[1,2,bs,rs])*x1 + conj(A[2,2,bs,rs])*x2 + conj(A[3,2,bs,rs])*x3
        y[3, alpha, b, r] = conj(A[1,3,bs,rs])*x1 + conj(A[2,3,bs,rs])*x2 + conj(A[3,3,bs,rs])*x3
    end
    return nothing
end

# Fused kernel: negative direction (a < 0, aa = -a)
# y(b,r) = U_aa(b,r) * x(bs,rs) * factor
function _cudakernel_fused_hop_neg!(y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    bs, rs = _shifted_idx(b, r, shift, blockinfo)
    fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    @inbounds for alpha = 1:4
        x1 = x[1, alpha, bs, rs] * fac
        x2 = x[2, alpha, bs, rs] * fac
        x3 = x[3, alpha, bs, rs] * fac
        y[1, alpha, b, r] = A[1,1,b,r]*x1 + A[1,2,b,r]*x2 + A[1,3,b,r]*x3
        y[2, alpha, b, r] = A[2,1,b,r]*x1 + A[2,2,b,r]*x2 + A[2,3,b,r]*x3
        y[3, alpha, b, r] = A[3,1,b,r]*x1 + A[3,2,b,r]*x2 + A[3,3,b,r]*x3
    end
    return nothing
end

const _BC_DEFAULT = (Int8(1), Int8(1), Int8(1), Int8(-1))

@inline function _dir_to_shift(a::Int)
    s1 = Int8(ifelse(abs(a) == 1, sign(a), 0))
    s2 = Int8(ifelse(abs(a) == 2, sign(a), 0))
    s3 = Int8(ifelse(abs(a) == 3, sign(a), 0))
    s4 = Int8(ifelse(abs(a) == 4, sign(a), 0))
    return (s1, s2, s3, s4)
end

export fused_hopping_gpu!
export fused_wilson_mul_gpu!
export batched_hopping_steps_gpu!
export batched_inv_hopping_gpu!
export get_shared_buf!

# ============================================================
#  Shared GPU buffer pool — layers 간 임시 버퍼 공유
#  (result_all, d_buf, ci_all, hop_temp 등 forward/backward 직후
#   unpack되어 재사용 가능한 버퍼를 공유하여 GPU 메모리 절약)
# ============================================================
const _shared_gpu_pool = Dict{Tuple{Symbol,Int}, CuArray{ComplexF64,5}}()

"""
    get_shared_buf!(key, NC, NG, bs, rs, n) → CuArray{ComplexF64,5}

Shared 5D GPU buffer. 동일 (key, n) 조합은 동일 CuArray를 반환.
PTLayer 인스턴스 간 임시 버퍼를 공유하여 GPU 메모리를 절약.
"""
function get_shared_buf!(key::Symbol, NC::Int, NG::Int, bs::Int, rs::Int, n::Int)
    k = (key, n)
    buf = get(_shared_gpu_pool, k, nothing)
    if buf === nothing || size(buf, 3) != bs || size(buf, 4) != rs
        buf = CUDA.zeros(ComplexF64, NC, NG, bs, rs, n)
        _shared_gpu_pool[k] = buf
    end
    return buf
end

# ============================================================
#  Fused Wilson Dirac kernel — replaces external mul!(y, D, x)
#  and mul!(y, D', x).  Zero allocations.
# ============================================================
#
# Convention (from LatticeDiracOperators):
#   D*x  = x - κ Σ_ν [ (r-γ_ν) U_ν(site) x(site+ν̂) +
#                        (r+γ_ν) U†_ν(site-ν̂) x(site-ν̂) ]
#   D†*x = x - κ Σ_ν [ (r+γ_ν) U_ν(site) x(site+ν̂) +
#                        (r-γ_ν) U†_ν(site-ν̂) x(site-ν̂) ]
#
# Gamma matrices (Euclidean, Dirac rep):
#   γ1: (1,4)=-i  (2,3)=-i  (3,2)=+i  (4,1)=+i
#   γ2: (1,4)=-1  (2,3)=+1  (3,2)=+1  (4,1)=-1
#   γ3: (1,3)=-i  (2,4)=+i  (3,1)=+i  (4,2)=-i
#   γ4: (1,3)=-1  (2,4)=-1  (3,1)=-1  (4,2)=-1
#
# With r=1, (1±γ_ν) has 2 nonzeros per row → spin projection.
# We hardcode these as inline functions for maximum performance.
# ============================================================

# ---- Spin projectors: each returns two 2-spinor components ----
# (1-γ_ν): used for forward hop in D, backward hop in D†
# (1+γ_ν): used for backward hop in D, forward hop in D†

# (1-γ₁)ψ:  row1 = ψ₁ + i·ψ₄,  row2 = ψ₂ + i·ψ₃,  row3 = ψ₃ - i·ψ₂,  row4 = ψ₄ - i·ψ₁
@inline _proj_m1(s1,s2,s3,s4) = (s1 + im*s4, s2 + im*s3, s3 - im*s2, s4 - im*s1)
# (1+γ₁)ψ:  row1 = ψ₁ - i·ψ₄,  row2 = ψ₂ - i·ψ₃,  row3 = ψ₃ + i·ψ₂,  row4 = ψ₄ + i·ψ₁
@inline _proj_p1(s1,s2,s3,s4) = (s1 - im*s4, s2 - im*s3, s3 + im*s2, s4 + im*s1)

# (1-γ₂)ψ:  row1 = ψ₁ + ψ₄,  row2 = ψ₂ - ψ₃,  row3 = ψ₃ - ψ₂,  row4 = ψ₄ + ψ₁
@inline _proj_m2(s1,s2,s3,s4) = (s1 + s4, s2 - s3, s3 - s2, s4 + s1)
# (1+γ₂)ψ:  row1 = ψ₁ - ψ₄,  row2 = ψ₂ + ψ₃,  row3 = ψ₃ + ψ₂,  row4 = ψ₄ - ψ₁
@inline _proj_p2(s1,s2,s3,s4) = (s1 - s4, s2 + s3, s3 + s2, s4 - s1)

# (1-γ₃)ψ:  row1 = ψ₁ + i·ψ₃,  row2 = ψ₂ - i·ψ₄,  row3 = ψ₃ - i·ψ₁,  row4 = ψ₄ + i·ψ₂
@inline _proj_m3(s1,s2,s3,s4) = (s1 + im*s3, s2 - im*s4, s3 - im*s1, s4 + im*s2)
# (1+γ₃)ψ:  row1 = ψ₁ - i·ψ₃,  row2 = ψ₂ + i·ψ₄,  row3 = ψ₃ + i·ψ₁,  row4 = ψ₄ - i·ψ₂
@inline _proj_p3(s1,s2,s3,s4) = (s1 - im*s3, s2 + im*s4, s3 + im*s1, s4 - im*s2)

# (1-γ₄)ψ:  row1 = ψ₁ + ψ₃,  row2 = ψ₂ + ψ₄,  row3 = ψ₃ + ψ₁,  row4 = ψ₄ + ψ₂
@inline _proj_m4(s1,s2,s3,s4) = (s1 + s3, s2 + s4, s3 + s1, s4 + s2)
# (1+γ₄)ψ:  row1 = ψ₁ - ψ₃,  row2 = ψ₂ - ψ₄,  row3 = ψ₃ - ψ₁,  row4 = ψ₄ - ψ₂
@inline _proj_p4(s1,s2,s3,s4) = (s1 - s3, s2 - s4, s3 - s1, s4 - s2)


function _cudakernel_fused_wilson!(
    yf     ::CuDeviceArray{ComplexF64, 4},   # output (3, 4, blocksize, rsize)
    xf     ::CuDeviceArray{ComplexF64, 4},   # input
    U1     ::CuDeviceArray{ComplexF64, 4},   # gauge μ=1 (3,3,bs,rs)
    U2     ::CuDeviceArray{ComplexF64, 4},
    U3     ::CuDeviceArray{ComplexF64, 4},
    U4     ::CuDeviceArray{ComplexF64, 4},
    κ      ::Float64,
    blockinfo,
    bc,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    is_dagger::Bool   # false = D, true = D†
)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    # Read x at current site
    @inbounds begin
        x1_1 = xf[1,1,b,r]; x2_1 = xf[2,1,b,r]; x3_1 = xf[3,1,b,r]
        x1_2 = xf[1,2,b,r]; x2_2 = xf[2,2,b,r]; x3_2 = xf[3,2,b,r]
        x1_3 = xf[1,3,b,r]; x2_3 = xf[2,3,b,r]; x3_3 = xf[3,3,b,r]
        x1_4 = xf[1,4,b,r]; x2_4 = xf[2,4,b,r]; x3_4 = xf[3,4,b,r]
    end

    # Accumulators for temp = Σ_ν contribution (per color, per spin)
    a1_1=zero(ComplexF64); a2_1=zero(ComplexF64); a3_1=zero(ComplexF64)
    a1_2=zero(ComplexF64); a2_2=zero(ComplexF64); a3_2=zero(ComplexF64)
    a1_3=zero(ComplexF64); a2_3=zero(ComplexF64); a3_3=zero(ComplexF64)
    a1_4=zero(ComplexF64); a2_4=zero(ComplexF64); a3_4=zero(ComplexF64)

    # ---- Process each direction ν = 1..4 ----
    for ν in 1:4
        Uν = ν == 1 ? U1 : (ν == 2 ? U2 : (ν == 3 ? U3 : U4))

        # ------ Forward hop: xplus = x(site+ν̂), temp1 = U_ν(site) * xplus ------
        shift_fwd = (Int8(ν==1), Int8(ν==2), Int8(ν==3), Int8(ν==4))
        bs_f, rs_f = _shifted_idx(b, r, shift_fwd, blockinfo)
        fac_f = _bc_factor(b, r, shift_fwd, blockinfo, bc, NX, NY, NZ, NT)

        @inbounds begin
            # Read shifted fermion x(site+ν̂)
            xp1_1 = xf[1,1,bs_f,rs_f]*fac_f; xp2_1 = xf[2,1,bs_f,rs_f]*fac_f; xp3_1 = xf[3,1,bs_f,rs_f]*fac_f
            xp1_2 = xf[1,2,bs_f,rs_f]*fac_f; xp2_2 = xf[2,2,bs_f,rs_f]*fac_f; xp3_2 = xf[3,2,bs_f,rs_f]*fac_f
            xp1_3 = xf[1,3,bs_f,rs_f]*fac_f; xp2_3 = xf[2,3,bs_f,rs_f]*fac_f; xp3_3 = xf[3,3,bs_f,rs_f]*fac_f
            xp1_4 = xf[1,4,bs_f,rs_f]*fac_f; xp2_4 = xf[2,4,bs_f,rs_f]*fac_f; xp3_4 = xf[3,4,bs_f,rs_f]*fac_f

            # U_ν(b,r) * xplus — for each spin component
            u11=Uν[1,1,b,r]; u12=Uν[1,2,b,r]; u13=Uν[1,3,b,r]
            u21=Uν[2,1,b,r]; u22=Uν[2,2,b,r]; u23=Uν[2,3,b,r]
            u31=Uν[3,1,b,r]; u32=Uν[3,2,b,r]; u33=Uν[3,3,b,r]
        end
        # temp1[color, spin] = U * xplus  (per spin)
        t1_1 = u11*xp1_1 + u12*xp2_1 + u13*xp3_1
        t2_1 = u21*xp1_1 + u22*xp2_1 + u23*xp3_1
        t3_1 = u31*xp1_1 + u32*xp2_1 + u33*xp3_1
        t1_2 = u11*xp1_2 + u12*xp2_2 + u13*xp3_2
        t2_2 = u21*xp1_2 + u22*xp2_2 + u23*xp3_2
        t3_2 = u31*xp1_2 + u32*xp2_2 + u33*xp3_2
        t1_3 = u11*xp1_3 + u12*xp2_3 + u13*xp3_3
        t2_3 = u21*xp1_3 + u22*xp2_3 + u23*xp3_3
        t3_3 = u31*xp1_3 + u32*xp2_3 + u33*xp3_3
        t1_4 = u11*xp1_4 + u12*xp2_4 + u13*xp3_4
        t2_4 = u21*xp1_4 + u22*xp2_4 + u23*xp3_4
        t3_4 = u31*xp1_4 + u32*xp2_4 + u33*xp3_4

        # Apply spin projector to temp1: D uses (r-γ_ν), D† uses (r+γ_ν)
        # Apply per color independently (spin projector acts on Dirac indices only)
        if !is_dagger
            # Forward hop in D: (1-γ_ν)
            if ν == 1
                p1_1,p1_2,p1_3,p1_4 = _proj_m1(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m1(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m1(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 2
                p1_1,p1_2,p1_3,p1_4 = _proj_m2(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m2(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m2(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 3
                p1_1,p1_2,p1_3,p1_4 = _proj_m3(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m3(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m3(t3_1,t3_2,t3_3,t3_4)
            else
                p1_1,p1_2,p1_3,p1_4 = _proj_m4(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m4(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m4(t3_1,t3_2,t3_3,t3_4)
            end
        else
            # Forward hop in D†: (1+γ_ν)
            if ν == 1
                p1_1,p1_2,p1_3,p1_4 = _proj_p1(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p1(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p1(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 2
                p1_1,p1_2,p1_3,p1_4 = _proj_p2(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p2(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p2(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 3
                p1_1,p1_2,p1_3,p1_4 = _proj_p3(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p3(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p3(t3_1,t3_2,t3_3,t3_4)
            else
                p1_1,p1_2,p1_3,p1_4 = _proj_p4(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p4(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p4(t3_1,t3_2,t3_3,t3_4)
            end
        end

        # Accumulate forward contribution (κ * projected)
        a1_1 += κ*p1_1; a2_1 += κ*p2_1; a3_1 += κ*p3_1
        a1_2 += κ*p1_2; a2_2 += κ*p2_2; a3_2 += κ*p3_2
        a1_3 += κ*p1_3; a2_3 += κ*p2_3; a3_3 += κ*p3_3
        a1_4 += κ*p1_4; a2_4 += κ*p2_4; a3_4 += κ*p3_4

        # ------ Backward hop: xminus = x(site-ν̂), temp2 = U†_ν(site-ν̂) * xminus ------
        shift_bwd = (-Int8(ν==1), -Int8(ν==2), -Int8(ν==3), -Int8(ν==4))
        bs_b, rs_b = _shifted_idx(b, r, shift_bwd, blockinfo)
        fac_b = _bc_factor(b, r, shift_bwd, blockinfo, bc, NX, NY, NZ, NT)

        @inbounds begin
            xm1_1 = xf[1,1,bs_b,rs_b]*fac_b; xm2_1 = xf[2,1,bs_b,rs_b]*fac_b; xm3_1 = xf[3,1,bs_b,rs_b]*fac_b
            xm1_2 = xf[1,2,bs_b,rs_b]*fac_b; xm2_2 = xf[2,2,bs_b,rs_b]*fac_b; xm3_2 = xf[3,2,bs_b,rs_b]*fac_b
            xm1_3 = xf[1,3,bs_b,rs_b]*fac_b; xm2_3 = xf[2,3,bs_b,rs_b]*fac_b; xm3_3 = xf[3,3,bs_b,rs_b]*fac_b
            xm1_4 = xf[1,4,bs_b,rs_b]*fac_b; xm2_4 = xf[2,4,bs_b,rs_b]*fac_b; xm3_4 = xf[3,4,bs_b,rs_b]*fac_b

            # U†_ν(site-ν̂): conj transpose of U_ν at shifted site
            v11=conj(Uν[1,1,bs_b,rs_b]); v12=conj(Uν[2,1,bs_b,rs_b]); v13=conj(Uν[3,1,bs_b,rs_b])
            v21=conj(Uν[1,2,bs_b,rs_b]); v22=conj(Uν[2,2,bs_b,rs_b]); v23=conj(Uν[3,2,bs_b,rs_b])
            v31=conj(Uν[1,3,bs_b,rs_b]); v32=conj(Uν[2,3,bs_b,rs_b]); v33=conj(Uν[3,3,bs_b,rs_b])
        end
        # temp2[color,spin] = U† * xminus
        q1_1 = v11*xm1_1 + v12*xm2_1 + v13*xm3_1
        q2_1 = v21*xm1_1 + v22*xm2_1 + v23*xm3_1
        q3_1 = v31*xm1_1 + v32*xm2_1 + v33*xm3_1
        q1_2 = v11*xm1_2 + v12*xm2_2 + v13*xm3_2
        q2_2 = v21*xm1_2 + v22*xm2_2 + v23*xm3_2
        q3_2 = v31*xm1_2 + v32*xm2_2 + v33*xm3_2
        q1_3 = v11*xm1_3 + v12*xm2_3 + v13*xm3_3
        q2_3 = v21*xm1_3 + v22*xm2_3 + v23*xm3_3
        q3_3 = v31*xm1_3 + v32*xm2_3 + v33*xm3_3
        q1_4 = v11*xm1_4 + v12*xm2_4 + v13*xm3_4
        q2_4 = v21*xm1_4 + v22*xm2_4 + v23*xm3_4
        q3_4 = v31*xm1_4 + v32*xm2_4 + v33*xm3_4

        # Apply spin projector to temp2: D uses (r+γ_ν), D† uses (r-γ_ν)
        if !is_dagger
            if ν == 1
                r1_1,r1_2,r1_3,r1_4 = _proj_p1(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p1(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p1(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 2
                r1_1,r1_2,r1_3,r1_4 = _proj_p2(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p2(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p2(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 3
                r1_1,r1_2,r1_3,r1_4 = _proj_p3(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p3(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p3(q3_1,q3_2,q3_3,q3_4)
            else
                r1_1,r1_2,r1_3,r1_4 = _proj_p4(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p4(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p4(q3_1,q3_2,q3_3,q3_4)
            end
        else
            if ν == 1
                r1_1,r1_2,r1_3,r1_4 = _proj_m1(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m1(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m1(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 2
                r1_1,r1_2,r1_3,r1_4 = _proj_m2(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m2(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m2(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 3
                r1_1,r1_2,r1_3,r1_4 = _proj_m3(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m3(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m3(q3_1,q3_2,q3_3,q3_4)
            else
                r1_1,r1_2,r1_3,r1_4 = _proj_m4(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m4(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m4(q3_1,q3_2,q3_3,q3_4)
            end
        end

        # Accumulate backward contribution
        a1_1 += κ*r1_1; a2_1 += κ*r2_1; a3_1 += κ*r3_1
        a1_2 += κ*r1_2; a2_2 += κ*r2_2; a3_2 += κ*r3_2
        a1_3 += κ*r1_3; a2_3 += κ*r2_3; a3_3 += κ*r3_3
        a1_4 += κ*r1_4; a2_4 += κ*r2_4; a3_4 += κ*r3_4
    end  # ν loop

    # y = x - accumulated
    @inbounds begin
        yf[1,1,b,r] = x1_1 - a1_1; yf[2,1,b,r] = x2_1 - a2_1; yf[3,1,b,r] = x3_1 - a3_1
        yf[1,2,b,r] = x1_2 - a1_2; yf[2,2,b,r] = x2_2 - a2_2; yf[3,2,b,r] = x3_2 - a3_2
        yf[1,3,b,r] = x1_3 - a1_3; yf[2,3,b,r] = x2_3 - a2_3; yf[3,3,b,r] = x3_3 - a3_3
        yf[1,4,b,r] = x1_4 - a1_4; yf[2,4,b,r] = x2_4 - a2_4; yf[3,4,b,r] = x3_4 - a3_4
    end
    return nothing
end

"""
    fused_wilson_mul_gpu!(y, x, U, κ, is_dagger)

Fused Wilson Dirac operator on GPU.  Zero allocations.
  - is_dagger=false: y = D*x
  - is_dagger=true:  y = D†*x
Replaces external LatticeDiracOperators mul!(y, D, x) which allocates
shift_fermion temporaries (~40 KB per call).
"""
function fused_wilson_mul_gpu!(y::WF3x1GPU, x::WF3x1GPU,
                               U::GPUgauge, κ::Float64, is_dagger::Bool)
    bi = x.blockinfo
    blocksize = bi.blocksize
    rsize     = bi.rsize
    bc = _BC_DEFAULT
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_wilson!(
        y.f, x.f, U[1].U, U[2].U, U[3].U, U[4].U,
        κ, bi, bc, NX, NY, NZ, NT, is_dagger)
end


"""
    fused_hopping_gpu!(V, x, Gauge, paths_various)

GPU hopping with fused kernels, double-buffering via f ↔ fshifted.
Per step: 3 kernel launches + 3 CUDA.@sync → 1 kernel + 0 sync.
"""
function fused_hopping_gpu!(V::AbstractVector{<:WF3x1GPU}, x::WF3x1GPU,
                            Gauge::GPUgauge, paths_various)
    blocksize = x.blockinfo.blocksize
    rsize     = x.blockinfo.rsize
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    bi = x.blockinfo
    bc = _BC_DEFAULT

    @inbounds for k in eachindex(V)
        copyx = V[k]
        steps = paths_various[k].steps

        # Initialize copyx from x (GPU memcpy, no kernel)
        copyto!(copyx.f, x.f)

        nsteps = length(steps)
        if nsteps == 0 || (nsteps == 1 && steps[1].f == 0)
            continue   # identity path
        end

        # Double-buffering: f ↔ fshifted, no intermediate sync
        buf_a = copyx.f
        buf_b = copyx.fshifted
        flip = false

        for step in steps
            a = step.f
            src = flip ? buf_b : buf_a
            dst = flip ? buf_a : buf_b

            if a > 0
                shift = _dir_to_shift(-a)
                CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_hop_pos!(
                    dst, Gauge[a].U, src, shift, bi, bc, NX, NY, NZ, NT)
            elseif a < 0
                aa = -a
                shift = _dir_to_shift(aa)
                CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_hop_neg!(
                    dst, Gauge[aa].U, src, shift, bi, bc, NX, NY, NZ, NT)
            else
                # a == 0: reset to original x
                copyto!(dst, x.f)
            end
            flip = !flip
        end

        # If result is in fshifted (odd steps), copy back to f
        if flip
            copyto!(copyx.f, copyx.fshifted)
        end
    end
end


# ============================================================
#  Batched multi-path hopping — 57 kernel launches → max_steps
# ============================================================

function _cudakernel_batched_hop_step!(
    dst, src, U1, U2, U3, U4,
    dirs,         # CuDeviceArray{Int8,2} (max_steps, npaths)
    nsteps_arr,   # CuDeviceArray{Int32,1} (npaths,)sync_W_to_gpu!
    step_idx,     # Int32
    npaths,       # Int32
    blockinfo, bc,
    NX::Int, NY::Int, NZ::Int, NT::Int
)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for pidx in Int32(1):npaths
        if step_idx > nsteps_arr[pidx]
            # Copy-through: path already done
            for g in 1:4
                dst[1,g,b,r,pidx] = src[1,g,b,r,pidx]
                dst[2,g,b,r,pidx] = src[2,g,b,r,pidx]
                dst[3,g,b,r,pidx] = src[3,g,b,r,pidx]
            end
        else
            d = Int(dirs[step_idx, pidx])
            if d > 0
                # positive direction: y(b,r) = U†_d(bs,rs) * x(bs,rs) * fac
                s1 = Int8(d==1 ? -1 : 0); s2 = Int8(d==2 ? -1 : 0)
                s3 = Int8(d==3 ? -1 : 0); s4 = Int8(d==4 ? -1 : 0)
                shift = (s1, s2, s3, s4)
                bs, rs = _shifted_idx(b, r, shift, blockinfo)
                fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
                Uν = d == 1 ? U1 : (d == 2 ? U2 : (d == 3 ? U3 : U4))
                u11=Uν[1,1,bs,rs]; u21=Uν[2,1,bs,rs]; u31=Uν[3,1,bs,rs]
                u12=Uν[1,2,bs,rs]; u22=Uν[2,2,bs,rs]; u32=Uν[3,2,bs,rs]
                u13=Uν[1,3,bs,rs]; u23=Uν[2,3,bs,rs]; u33=Uν[3,3,bs,rs]
                for alpha in 1:4
                    x1 = src[1,alpha,bs,rs,pidx]*fac
                    x2 = src[2,alpha,bs,rs,pidx]*fac
                    x3 = src[3,alpha,bs,rs,pidx]*fac
                    dst[1,alpha,b,r,pidx] = conj(u11)*x1 + conj(u21)*x2 + conj(u31)*x3
                    dst[2,alpha,b,r,pidx] = conj(u12)*x1 + conj(u22)*x2 + conj(u32)*x3
                    dst[3,alpha,b,r,pidx] = conj(u13)*x1 + conj(u23)*x2 + conj(u33)*x3
                end
            elseif d < 0
                # negative direction: y(b,r) = U_|d|(b,r) * x(bs,rs) * fac
                aa = -d
                s1 = Int8(aa==1 ? 1 : 0); s2 = Int8(aa==2 ? 1 : 0)
                s3 = Int8(aa==3 ? 1 : 0); s4 = Int8(aa==4 ? 1 : 0)
                shift = (s1, s2, s3, s4)
                bs, rs = _shifted_idx(b, r, shift, blockinfo)
                fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
                Uν = aa == 1 ? U1 : (aa == 2 ? U2 : (aa == 3 ? U3 : U4))
                u11=Uν[1,1,b,r]; u12=Uν[1,2,b,r]; u13=Uν[1,3,b,r]
                u21=Uν[2,1,b,r]; u22=Uν[2,2,b,r]; u23=Uν[2,3,b,r]
                u31=Uν[3,1,b,r]; u32=Uν[3,2,b,r]; u33=Uν[3,3,b,r]
                for alpha in 1:4
                    x1 = src[1,alpha,bs,rs,pidx]*fac
                    x2 = src[2,alpha,bs,rs,pidx]*fac
                    x3 = src[3,alpha,bs,rs,pidx]*fac
                    dst[1,alpha,b,r,pidx] = u11*x1 + u12*x2 + u13*x3
                    dst[2,alpha,b,r,pidx] = u21*x1 + u22*x2 + u23*x3
                    dst[3,alpha,b,r,pidx] = u31*x1 + u32*x2 + u33*x3
                end
            else
                # d == 0: identity step — copy through
                for g in 1:4
                    dst[1,g,b,r,pidx] = src[1,g,b,r,pidx]
                    dst[2,g,b,r,pidx] = src[2,g,b,r,pidx]
                    dst[3,g,b,r,pidx] = src[3,g,b,r,pidx]
                end
            end
        end
    end
    return nothing
end

"""
    batched_hopping_steps_gpu!(hopped_all, temp_5d, x_fields, Gauge,
                                dirs_gpu, nsteps_gpu, max_steps, npaths,
                                blocksize, rsize)

25개 fused_hopping_gpu! (57 커널) → max_steps 커널.
Double-buffering: hopped_all ↔ temp_5d, step별 1 커널.
"""
function batched_hopping_steps_gpu!(hopped_all, temp_5d, x_fields,
                                    Gauge, dirs_gpu, nsteps_gpu, max_steps, npaths,
                                    blocksize, rsize)
    bi = x_fields[1].blockinfo
    bc = _BC_DEFAULT
    NX = x_fields[1].NX; NY = x_fields[1].NY
    NZ = x_fields[1].NZ; NT = x_fields[1].NT

    # 1) Pack all x[i] → hopped_all
    pack_fermions!(hopped_all, x_fields)

    # 2) Step-by-step batched hop with double-buffering
    buf_a = hopped_all
    buf_b = temp_5d
    for s in Int32(1):Int32(max_steps)
        CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_hop_step!(
            buf_b, buf_a,
            Gauge[1].U, Gauge[2].U, Gauge[3].U, Gauge[4].U,
            dirs_gpu, nsteps_gpu, s, Int32(npaths),
            bi, bc, NX, NY, NZ, NT)
        buf_a, buf_b = buf_b, buf_a
    end

    # 3) After max_steps swaps: if even, buf_a=hopped_all; if odd, buf_a=temp_5d
    if isodd(max_steps)
        copyto!(hopped_all, temp_5d)
    end
end

"""
    batched_inv_hopping_gpu!(ci_all, temp_5d, Gauge,
                              inv_dirs_gpu, nsteps_gpu, max_steps, npaths,
                              blocksize, rsize, bi, NX, NY, NZ, NT)

DeltaRule! 역경로 hop을 batched로 수행.
ci_all을 in-place double-buffering으로 처리 (ci_all ↔ temp_5d).
결과는 ci_all에 저장됨.
"""
function batched_inv_hopping_gpu!(ci_all, temp_5d, Gauge,
                                   inv_dirs_gpu, nsteps_gpu, max_steps, npaths,
                                   blocksize, rsize, bi, NX, NY, NZ, NT)
    bc = _BC_DEFAULT

    buf_a = ci_all
    buf_b = temp_5d
    for s in Int32(1):Int32(max_steps)
        CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_hop_step!(
            buf_b, buf_a,
            Gauge[1].U, Gauge[2].U, Gauge[3].U, Gauge[4].U,
            inv_dirs_gpu, nsteps_gpu, s, Int32(npaths),
            bi, bc, NX, NY, NZ, NT)
        buf_a, buf_b = buf_b, buf_a
    end

    if isodd(max_steps)
        copyto!(ci_all, temp_5d)
    end
end


end # module gpu