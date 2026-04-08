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
        result[idx] = Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(M_cpu[:,:,idx]))
    end
    return result
end

export Matmul_batch







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

export _build_block_weight_gpu


end # module gpu