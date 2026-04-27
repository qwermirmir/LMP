module elementry_base


export Hopping_operator
export Onestep
export Paths
export PTC
export CPTC
export Random_SpinM
export Spin_matrix
export inner_product
export norm2
export reverse_paths
export Clifford 
export mean
export standardization
export minmax
export g2vector_generator!
export substitute_fermion!
export blockize
export unblockize
export CloverDiracOperator
export compute_Fmunu
export refresh_clover!

# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
import LinearAlgebra: mul!   # ← 이 줄이 중요!
using CUDA
using StaticArrays

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}


const Dirac_operator =LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}
const DagDirac_operator =LatticeDiracOperators.Dirac_operators.Adjoint_Wilson_operator{LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}
const DdagD_operator =LatticeDiracOperators.Dirac_operators.DdagD_Wilson_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}


import Base: getindex, setindex!, size, length, ndims, broadcastable
export AbstractSpinMatrix,mean,spincolortrace,addmul!


abstract type AbstractSpinMatrix <: AbstractArray{ComplexF64,2} end

mutable struct Spin_matrix <: AbstractSpinMatrix
    sm::Matrix{ComplexF64}
end


# 배열 크기 반환
size(S::Spin_matrix) = size(S.sm)  

# 배열 원소 개수 반환
length(S::Spin_matrix) = length(S.sm)  

# 차원 수 반환 (2차원 행렬이므로 2)
ndims(S::Spin_matrix) = 2

ndims(::Type{WF3x1}) = 6

# Overload Base.adjoint for Spin_matrix
Base.adjoint(S::Spin_matrix) = Spin_matrix(adjoint(S.sm))

import Base: broadcastable

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



function Base.:+(a::WF3x1,b::WF3x1)

    y=deepcopy(b)



    axpby!(1,a,1,y)

    return y
end 

function Base.:+(a::Vector{WF3x1},b::Vector{WF3x1})

    c=deepcopy(b)

    for i in 1:length(b)
        c[i]=a[i]+b[i]
    end

    return c
end



function Base.:-(a::WF3x1,b::WF3x1)

    y=deepcopy(b)



    axpby!(1,a,-1,y)

    return y
end 

function Base.:*(g::AbstractGaugefields{NC,Dim},a::WF3x1) where {NC,Dim,T<:AbstractGaugefields}


    b=similar(a)

    mul!(b,g,a)
    
    return b
end 


function Base.:*(a::Number,b::Spin_matrix)
    
    c=deepcopy(b)

    c.sm=a*b.sm

    return c
end




function Base.:*(M::Spin_matrix,x::WF3x1)
    
    
    f=deepcopy(x)
    
    NX = f.NX
    NY = f.NY
    NZ = f.NZ
    NT = f.NT
    NC = f.NC



    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for ic = 1:NC
                        e1 = f[ic, ix, iy, iz, it, 1]
                        e2 = f[ic, ix, iy, iz, it, 2]
                        e3 = f[ic, ix, iy, iz, it, 3]
                        e4 = f[ic, ix, iy, iz, it, 4]

                        f[ic, ix, iy, iz, it, 1] =
                            M[1, 1] * e1 + M[1, 2] * e2 + M[1, 3] * e3 + M[1, 4] * e4
                        f[ic, ix, iy, iz, it, 2] =
                            M[2, 1] * e1 + M[2, 2] * e2 + M[2, 3] * e3 + M[2, 4] * e4
                        f[ic, ix, iy, iz, it, 3] =
                            M[3, 1] * e1 + M[3, 2] * e2 + M[3, 3] * e3 + M[3, 4] * e4
                        f[ic, ix, iy, iz, it, 4] =
                            M[4, 1] * e1 + M[4, 2] * e2 + M[4, 3] * e3 + M[4, 4] * e4
                    end
                end
            end
        end
    end

    return f
end


function mul!(f::WF3x1, M::Spin_matrix, x::WF3x1)
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT; NC = x.NC

    # M 요소를 미리 로컬 변수에 빼놓기 (인덱싱 비용 줄이기)
    m11 = M[1,1]; m12 = M[1,2]; m13 = M[1,3]; m14 = M[1,4]
    m21 = M[2,1]; m22 = M[2,2]; m23 = M[2,3]; m24 = M[2,4]
    m31 = M[3,1]; m32 = M[3,2]; m33 = M[3,3]; m34 = M[3,4]
    m41 = M[4,1]; m42 = M[4,2]; m43 = M[4,3]; m44 = M[4,4]

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for ic = 1:NC
                        e1 = x[ic, ix, iy, iz, it, 1]
                        e2 = x[ic, ix, iy, iz, it, 2]
                        e3 = x[ic, ix, iy, iz, it, 3]
                        e4 = x[ic, ix, iy, iz, it, 4]

                        f[ic, ix, iy, iz, it, 1] =
                            m11 * e1 + m12 * e2 + m13 * e3 + m14 * e4
                        f[ic, ix, iy, iz, it, 2] =
                            m21 * e1 + m22 * e2 + m23 * e3 + m24 * e4
                        f[ic, ix, iy, iz, it, 3] =
                            m31 * e1 + m32 * e2 + m33 * e3 + m34 * e4
                        f[ic, ix, iy, iz, it, 4] =
                            m41 * e1 + m42 * e2 + m43 * e3 + m44 * e4
                    end
                end
            end
        end
    end

    return f
end


function addmul!(f::WF3x1, M::AbstractSpinMatrix, x::WF3x1)
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT; NC = x.NC

    # M 요소를 미리 로컬 변수에 빼놓기 (인덱싱 비용 줄이기)
    m11 = M[1,1]; m12 = M[1,2]; m13 = M[1,3]; m14 = M[1,4]
    m21 = M[2,1]; m22 = M[2,2]; m23 = M[2,3]; m24 = M[2,4]
    m31 = M[3,1]; m32 = M[3,2]; m33 = M[3,3]; m34 = M[3,4]
    m41 = M[4,1]; m42 = M[4,2]; m43 = M[4,3]; m44 = M[4,4]

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for ic = 1:NC
                        e1 = x[ic, ix, iy, iz, it, 1]
                        e2 = x[ic, ix, iy, iz, it, 2]
                        e3 = x[ic, ix, iy, iz, it, 3]
                        e4 = x[ic, ix, iy, iz, it, 4]

                        f[ic, ix, iy, iz, it, 1] +=
                            m11 * e1 + m12 * e2 + m13 * e3 + m14 * e4
                        f[ic, ix, iy, iz, it, 2] +=
                            m21 * e1 + m22 * e2 + m23 * e3 + m24 * e4
                        f[ic, ix, iy, iz, it, 3] +=
                            m31 * e1 + m32 * e2 + m33 * e3 + m34 * e4
                        f[ic, ix, iy, iz, it, 4] +=
                            m41 * e1 + m42 * e2 + m43 * e3 + m44 * e4
                    end
                end
            end
        end
    end

    return f
end







function Base.:*(b::Spin_matrix,a::Vector{WF3x1})

    c=deepcopy(a)


    for i in 1:length(a)
        c[i]=b*a[i]
    end

    return c
end

function Base.:*(a::Vector{Spin_matrix},b::Vector{WF3x1})


    c=deepcopy(b)

    for i in 1:length(b)
        c[i]=a[i]*b[i]
    end

    return c
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



import Base.Broadcast: broadcast!

function broadcast!(::typeof(/), x::WF3x1, y::Number)
    for ic in 1:x.NC, it in 1:x.NT, iz in 1:x.NZ, iy in 1:x.NY, ix in 1:x.NX
        for i in 1:4
            x[ic, ix, iy, iz, it, i] /= y
        end
    end
    return x
end

function Base.:*(D::Dirac_operator,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end

function Base.:*(D::DagDirac_operator,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end

function Base.:*(x::WF3x1,D::Dirac_operator)
    cx=conj(x)

    y=D'*cx

    return conj(y)
end 


function Base.:*(D::DdagD_operator,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end




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

function Base.conj(WF3x1)
    f=deepcopy(WF3x1)
    NX = f.NX
    NY = f.NY
    NZ = f.NZ
    NT = f.NT
    NC = f.NC
    NG = f.NG

    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            f[ic, ix, iy, iz, it, ig] = conj(f[ic, ix, iy, iz, it, ig])
                        end
                    end
                end
            end
        end
    end
    return f
end




function Base.:/(f::WF3x1,c::Number)
    g=similar(f)

    axpby!(1/c,f,0,g)

    return g
end







function Base.conj(W::Spin_matrix)
    return Spin_matrix(conj(W.sm))
end







function inner_product(x::WF3x1,f::WF3x1)
    
    k=0


    k=dot(x,f)


    return k
end


function norm2(x::WF3x1)
    return inner_product(x,x)
end


# CPU 버전 substitute_fermion! - inplace copy (메모리 할당 없음)
function substitute_fermion!(dest::WF3x1CPU, src::WF3x1CPU)
    @assert size(dest.f) == size(src.f) "dest와 src의 크기가 같아야 합니다."
    copyto!(dest.f, src.f)
    return dest
end


# ============================================================
# Clover Dirac operator  (Sheikholeslami-Wohlert)
# D_cl = D_Wilson + (i*c_SW*κ) Σ_{μ<ν} σ_μν F̂_μν
# Here σ_μν = (i/2)[γ_μ,γ_ν] and F̂_μν = (Q_μν - Q_μν†)/8.
# ============================================================

export CloverDiracOperator

# --- 좌표 한 칸 이동 (periodic BC) ---
@inline function _pbc(x::Int, shift::Int, N::Int)
    xs = x + shift
    xs > N && return xs - N
    xs < 1 && return xs + N
    return xs
end

# --- 한 사이트의 3×3 게이지 링크를 SMatrix로 읽기 ---
@inline function _read_link(Ug, ix, iy, iz, it)
    @inbounds SMatrix{3,3,ComplexF64,9}(
        Ug[1,1,ix,iy,iz,it], Ug[2,1,ix,iy,iz,it], Ug[3,1,ix,iy,iz,it],
        Ug[1,2,ix,iy,iz,it], Ug[2,2,ix,iy,iz,it], Ug[3,2,ix,iy,iz,it],
        Ug[1,3,ix,iy,iz,it], Ug[2,3,ix,iy,iz,it], Ug[3,3,ix,iy,iz,it]
    )
end

# --- 주어진 방향으로 사이트 인덱스를 ±1 이동 ---
@inline function _site_shift(ix, iy, iz, it, mu, sign, NX, NY, NZ, NT)
    if mu == 1
        return (_pbc(ix, sign, NX), iy, iz, it)
    elseif mu == 2
        return (ix, _pbc(iy, sign, NY), iz, it)
    elseif mu == 3
        return (ix, iy, _pbc(iz, sign, NZ), it)
    else # mu == 4
        return (ix, iy, iz, _pbc(it, sign, NT))
    end
end

"""
    compute_Fmunu(U, mu, nu)

Clover field strength tensor F̂_μν(n)을 계산.
반에르미트 무대각 (traceless anti-hermitian) 결과를 반환.

Returns: Array{ComplexF64, 6} of shape (3, 3, NX, NY, NZ, NT)
"""
function compute_Fmunu(U::Vector, mu::Int, nu::Int)
    Umu = U[mu].U
    Unu = U[nu].U
    NX = size(Umu, 3); NY = size(Umu, 4)
    NZ = size(Umu, 5); NT = size(Umu, 6)

    Fmn = zeros(ComplexF64, 3, 3, NX, NY, NZ, NT)

    @inbounds for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX
        n  = (ix, iy, iz, it)
        # 이웃 사이트 인덱스
        n_pmu       = _site_shift(ix,iy,iz,it, mu, +1, NX,NY,NZ,NT)
        n_pnu       = _site_shift(ix,iy,iz,it, nu, +1, NX,NY,NZ,NT)
        n_mmu       = _site_shift(ix,iy,iz,it, mu, -1, NX,NY,NZ,NT)
        n_mnu       = _site_shift(ix,iy,iz,it, nu, -1, NX,NY,NZ,NT)
        n_pmu_mnu   = _site_shift(n_pmu..., nu, -1, NX,NY,NZ,NT)
        n_mmu_pnu   = _site_shift(n_mmu..., nu, +1, NX,NY,NZ,NT)
        n_mmu_mnu   = _site_shift(n_mmu..., nu, -1, NX,NY,NZ,NT)

        # 링크 읽기
        U_mu_n       = _read_link(Umu, n...)
        U_nu_n       = _read_link(Unu, n...)
        U_nu_pmu     = _read_link(Unu, n_pmu...)
        U_mu_pnu     = _read_link(Umu, n_pnu...)
        U_mu_mmu     = _read_link(Umu, n_mmu...)
        U_nu_mnu     = _read_link(Unu, n_mnu...)
        U_mu_mmu_pnu = _read_link(Umu, n_mmu_pnu...)
        U_nu_mmu     = _read_link(Unu, n_mmu...)
        U_nu_mmu_mnu = _read_link(Unu, n_mmu_mnu...)
        U_mu_mmu_mnu = _read_link(Umu, n_mmu_mnu...)
        U_nu_mnu     = _read_link(Unu, n_mnu...)
        U_mu_mnu     = _read_link(Umu, n_mnu...)
        U_nu_pmu_mnu = _read_link(Unu, n_pmu_mnu...)

        # ---- 4개의 plaquette (clover leaf) ----
        # P1 (+μ,+ν): U_μ(n) U_ν(n+μ̂) U_μ†(n+ν̂) U_ν†(n)
        P1 = U_mu_n * U_nu_pmu * U_mu_pnu' * U_nu_n'

        # P2 (+ν,−μ): U_ν(n) U_μ†(n−μ̂+ν̂) U_ν†(n−μ̂) U_μ(n−μ̂)
        P2 = U_nu_n * U_mu_mmu_pnu' * U_nu_mmu' * U_mu_mmu

        # P3 (−μ,−ν): U_μ†(n−μ̂) U_ν†(n−μ̂−ν̂) U_μ(n−μ̂−ν̂) U_ν(n−ν̂)
        P3 = U_mu_mmu' * U_nu_mmu_mnu' * U_mu_mmu_mnu * U_nu_mnu

        # P4 (−ν,+μ): U_ν†(n−ν̂) U_μ(n−ν̂) U_ν(n+μ̂−ν̂) U_μ†(n)
        P4 = U_nu_mnu' * U_mu_mnu * U_nu_pmu_mnu * U_mu_n'

        # Q = P1 + P2 + P3 + P4   (clover sum)
        Q = P1 + P2 + P3 + P4

        # F̂ = (Q − Q†)/8  (anti-hermitian)
        F_site = (Q - Q') / 8

        # traceless projection: F̂ -= (tr F̂ / 3) I
        tr_F = (F_site[1,1] + F_site[2,2] + F_site[3,3]) / 3
        F_site = F_site - tr_F * SMatrix{3,3}(1.0+0im,0,0, 0,1.0+0im,0, 0,0,1.0+0im)

        # 결과 저장
        for jc = 1:3, ic = 1:3
            Fmn[ic, jc, ix, iy, iz, it] = F_site[ic, jc]
        end
    end
    return Fmn
end

# σ_μν 행렬 6개를 (μ,ν) = (1,2),(1,3),(1,4),(2,3),(2,4),(3,4) 순서로 생성
# LatticeDiracOperators의 γ convention (Euclidean chiral basis)을 사용
function _build_sigma_munu()
    # LatticeDiracOperators mk_gamma() 의 γ 행렬 (Euclidean, hermitian, {γ_μ,γ_ν}=2δ_μν)
    g = Vector{Matrix{ComplexF64}}(undef, 4)

    g[1] = ComplexF64[0 0 0 -im;  0 0 -im 0;  0 im 0 0;  im 0 0 0]
    g[2] = ComplexF64[0 0 0 -1;   0 0 1 0;    0 1 0 0;   -1 0 0 0]
    g[3] = ComplexF64[0 0 -im 0;  0 0 0 im;   im 0 0 0;  0 -im 0 0]
    g[4] = ComplexF64[0 0 -1 0;   0 0 0 -1;   -1 0 0 0;  0 -1 0 0]

    sigma = Vector{Matrix{ComplexF64}}()
    for mu = 1:4, nu = (mu+1):4
        # σ_μν = (i/2)[γ_μ, γ_ν] = i γ_μ γ_ν  (since {γ_μ,γ_ν}=2δ_μν, μ≠ν → γ_μγ_ν=-γ_νγ_μ)
        s = im * g[mu] * g[nu]
        push!(sigma, s)
    end
    return sigma   # length 6
end

# (μ,ν) pair → σ 배열의 인덱스
@inline function _sigma_index(mu::Int, nu::Int)
    # (1,2)→1, (1,3)→2, (1,4)→3, (2,3)→4, (2,4)→5, (3,4)→6
    mu < nu || error("mu must be < nu")
    return (mu - 1) * (7 - mu) ÷ 2 + (nu - mu)
end

"""
    CloverDiracOperator(D_W, U, csw)

Clover-improved Wilson Dirac operator.
- `D_W`: Wilson Dirac operator (from LatticeDiracOperators)
- `U`:   gauge field vector `[U_1, U_2, U_3, U_4]`
- `csw`: Sheikholeslami-Wohlert coefficient c_SW (default 1.0)
"""
mutable struct CloverDiracOperator
    D_W                                    # Wilson (or wrapper, e.g. DdagD) Dirac operator
    U::Vector                              # gauge links
    csw::Float64                           # c_SW
    κ::Float64                             # hopping parameter (cached from D_W)
    Fmunu::Vector{Array{ComplexF64, 6}}    # 6 field strength tensors
    sigma::Vector{Matrix{ComplexF64}}      # 6 σ_μν matrices (4×4)

    function CloverDiracOperator(D_W, U::Vector, csw::Float64 = 1.0; κ::Union{Nothing,Real}=nothing)
        # σ_μν 계산
        sigma = _build_sigma_munu()

        # F̂_μν 계산 (6 planes)
        Fmunu = Vector{Array{ComplexF64, 6}}()
        for mu = 1:4, nu = (mu+1):4
            push!(Fmunu, compute_Fmunu(U, mu, nu))
        end

        κval = κ === nothing ? _extract_kappa(D_W) : Float64(κ)
        return new(D_W, U, csw, κval, Fmunu, sigma)
    end
end

# DdagD_Wilson_operator 등 wrapper에서도 κ를 꺼내올 수 있게 재귀적으로 탐색
function _extract_kappa(D)
    if hasproperty(D, :κ)
        return Float64(getfield(D, :κ))
    elseif hasproperty(D, :dirac)        # DdagD_Wilson_operator → inner Wilson
        return _extract_kappa(getfield(D, :dirac))
    elseif hasproperty(D, :D)            # 기타 wrapper
        return _extract_kappa(getfield(D, :D))
    else
        error("CloverDiracOperator: cannot find `κ` on $(typeof(D)). " *
              "Pass κ keyword explicitly: CloverDiracOperator(D, U, csw; κ=...).")
    end
end

# ---- Clover term 적용: y += (i*c_SW*κ) Σ_{μ<ν} σ_μν ⊗ F̂_μν * x ----
function _apply_clover_term!(y::WF3x1CPU, x::WF3x1CPU, Dcl::CloverDiracOperator)
    csw = Dcl.csw
    κ   = Dcl.κ
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    coeff = im * csw * κ

    @inbounds for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX
        # 이 사이트의 fermion 성분 읽기: x_buf[dirac, color]
        x_buf = MMatrix{4, 3, ComplexF64, 12}(undef)
        for ig = 1:4, ic = 1:3
            x_buf[ig, ic] = x[ic, ix, iy, iz, it, ig]
        end

        # accum[dirac, color] — clover 기여분 축적
        accum = @MMatrix zeros(ComplexF64, 4, 3)

        for p = 1:6   # 6 planes
            sigma_p = Dcl.sigma[p]
            Fp      = Dcl.Fmunu[p]

            # temp[α, a] = Σ_{β,b} σ[α,β] * F[a,b] * x[β,b]
            for alpha = 1:4, a = 1:3
                val = zero(ComplexF64)
                for beta = 1:4
                    s_ab = sigma_p[alpha, beta]
                    iszero(s_ab) && continue
                    for b = 1:3
                        val += s_ab * Fp[a, b, ix, iy, iz, it] * x_buf[beta, b]
                    end
                end
                accum[alpha, a] += val
            end
        end

        # y에 clover 기여분 추가
        for ig = 1:4, ic = 1:3
            y[ic, ix, iy, iz, it, ig] += coeff * accum[ig, ic]
        end
    end
    return nothing
end

# mul!(y, D_cl, x):  y = D_cl * x
function LinearAlgebra.mul!(y::WF3x1CPU, Dcl::CloverDiracOperator, x::WF3x1CPU)
    # 1) Wilson part
    mul!(y, Dcl.D_W, x)
    # 2) Clover part
    _apply_clover_term!(y, x, Dcl)
    return y
end

function Base.:*(Dcl::CloverDiracOperator, x::WF3x1)
    y = deepcopy(x)
    mul!(y, Dcl, x)
    return y
end

# F̂_μν 재계산 (게이지 업데이트 후 호출)
function refresh_clover!(Dcl::CloverDiracOperator)
    idx = 0
    for mu = 1:4, nu = (mu+1):4
        idx += 1
        Dcl.Fmunu[idx] = compute_Fmunu(Dcl.U, mu, nu)
    end
    return nothing
end


struct path_number
    f::Int64

    function path_number(x::Int64)
        # 허용되는 값인지 확인
        if x in (0, 1, 2, 3, 4, -1, -2, -3, -4)
            return new(x)
        else
            error("path_number는 0,1,2,3,4,-1,-2,-3,-4 중 하나만 허용합니다.")
        end
    end
end

struct Path
    steps::Vector{path_number}
end

function Path(ints::Vector{Int})
    # ints: [1, 2, -4, ...]
    # 변환 -> path_number(...) 배열
    pn_array = [path_number(i) for i in ints]
    return Path(pn_array)
end



struct Paths
    various::Vector{Path}

    # ✅ 여러 개의 개별 벡터를 받음
    function Paths(paths::Vararg{Vector{Int64}})  # 개별 벡터를 받음
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    # ✅ 기존 방식: 한 번에 묶인 Vector{Vector{Int64}} 받음
    function Paths(paths::Vector{Vector{Int64}})
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    # 기존 Path 리스트를 받는 생성자 유지
    function Paths(various::Vector{Path})
        return new(various)
    end
end

length(ps::Paths) = length(ps.various)



# ✅ 개별 Path를 리스트 형태로 출력
function Base.show(io::IO, p::Path)
    print(io, [step.f for step in p.steps])
end

# ✅ 개별 Path의 steps를 Int 리스트로 반환
function Base.getindex(paths::Paths, i::Int)
    return [step.f for step in paths.various[i].steps]
end

function Onestep()
    dirs = (0, 1, 2, 3, 4, -1, -2, -3, -4)
    single_steps = [[d] for d in dirs]  # 각 값을 리스트로 감싸기
    return Paths(single_steps)  # 가변 인자(...)를 제거
end




function reverse_path(path::Path)
    reversed_steps = [path_number(-step.f) for step in reverse(path.steps)]
    return Path(reversed_steps)
end

function reverse_paths(paths::Paths)
    reversed_various = [reverse_path(path) for path in paths.various]
    return Paths([[step.f for step in p.steps] for p in reversed_various])  # 개별 원소에 `.f` 적용
end




mutable struct Hopping_operator{G<:AbstractGaugefields}
    Gauge::Vector{G}
    P::Paths
end



function (hop::Hopping_operator)(x::WF3x1CPU)

    U = hop.Gauge
    shifts = Vector{WF3x1}()

    for k in 1:length(hop.P.various)
        # Ucopy 삭제 - U는 읽기만 하므로 deepcopy 불필요
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)  # Ucopy 대신 U 사용

                mul!(copyx, Uminus',shift_x)


            elseif a < 0

                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)  # Ucopy 대신 U 사용
            
            elseif a == 0

                copyx = deepcopy(x)
                
            end
        end

        push!(shifts, copyx)

    end

    return shifts
end

function (hop::Hopping_operator)(x::Vector{WF3x1CPU})

    U = hop.Gauge
    shifts = Vector{WF3x1}()

    for k in 1:length(hop.P.various)
        # Ucopy 삭제 - U는 읽기만 하므로 deepcopy 불필요
        copyx = deepcopy(x[k])
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)  # Ucopy 대신 U 사용

                mul!(copyx, Uminus',shift_x)
            
            elseif a < 0
                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)  # Ucopy 대신 U 사용
            elseif a == 0
                copyx = deepcopy(x[k])
                
            end
        end

        push!(shifts, copyx)

    end
    return shifts
end


# ============================================================
# Zero-allocation hop step 함수들
# shift_fermion/shift_U의 래퍼 구조체 할당을 완전히 제거
# ============================================================

const _bc_default = Int8[1, 1, 1, -1]

# shifted_fermion! 인라인 버전: copyx.f → copyx.fshifted (BC 적용)
function _shifted_fermion_inplace!(F::WF3x1CPU, shift::NTuple{4,Int})
    bc = _bc_default
    NX = F.NX; NY = F.NY; NZ = F.NZ; NT = F.NT
    sf = F.f; fsh = F.fshifted

    @inbounds for ig = 1:4
        for it = 1:NT
            it_s = it + shift[4]
            inside_up = it_s > NT; inside_down = it_s < 1
            factor_t = ifelse(inside_up || inside_down, Int(bc[4]), 1)
            it_s += ifelse(inside_up, -NT, 0)
            it_s += ifelse(inside_down, NT, 0)
            for iz = 1:NZ
                iz_s = iz + shift[3]
                inside_up = iz_s > NZ; inside_down = iz_s < 1
                factor_z = ifelse(inside_up || inside_down, Int(bc[3]), 1)
                iz_s += ifelse(inside_up, -NZ, 0)
                iz_s += ifelse(inside_down, NZ, 0)
                for iy = 1:NY
                    iy_s = iy + shift[2]
                    inside_up = iy_s > NY; inside_down = iy_s < 1
                    factor_y = ifelse(inside_up || inside_down, Int(bc[2]), 1)
                    iy_s += ifelse(inside_up, -NY, 0)
                    iy_s += ifelse(inside_down, NY, 0)
                    for ix = 1:NX
                        ix_s = ix + shift[1]
                        inside_up = ix_s > NX; inside_down = ix_s < 1
                        factor_x = ifelse(inside_up || inside_down, Int(bc[1]), 1)
                        ix_s += ifelse(inside_up, -NX, 0)
                        ix_s += ifelse(inside_down, NX, 0)
                        factor = factor_x * factor_y * factor_z * factor_t
                        @simd for ic = 1:3
                            fsh[ic, ix, iy, iz, it, ig] = factor * sf[ic, ix_s, iy_s, iz_s, it_s, ig]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# Forward step: copyx[site] = U†(site-μ̂) * copyx_old(site-μ̂)
# shift_U 제거: 게이지도 같은 shifted 인덱스에서 직접 읽음
function _hop_step_forward!(copyx::WF3x1CPU, U_mu, mu::Int)
    bc = _bc_default
    NX = copyx.NX; NY = copyx.NY; NZ = copyx.NZ; NT = copyx.NT
    sf = copyx.f; fsh = copyx.fshifted
    Ug = U_mu.U

    shift1 = (mu == 1) ? -1 : 0
    shift2 = (mu == 2) ? -1 : 0
    shift3 = (mu == 3) ? -1 : 0
    shift4 = (mu == 4) ? -1 : 0

    @inbounds for ig = 1:4
        for it = 1:NT
            it_s = it + shift4
            inside_up = it_s > NT; inside_down = it_s < 1
            factor_t = ifelse(inside_up || inside_down, Int(bc[4]), 1)
            it_s += ifelse(inside_up, -NT, 0)
            it_s += ifelse(inside_down, NT, 0)
            for iz = 1:NZ
                iz_s = iz + shift3
                inside_up = iz_s > NZ; inside_down = iz_s < 1
                factor_z = ifelse(inside_up || inside_down, Int(bc[3]), 1)
                iz_s += ifelse(inside_up, -NZ, 0)
                iz_s += ifelse(inside_down, NZ, 0)
                for iy = 1:NY
                    iy_s = iy + shift2
                    inside_up = iy_s > NY; inside_down = iy_s < 1
                    factor_y = ifelse(inside_up || inside_down, Int(bc[2]), 1)
                    iy_s += ifelse(inside_up, -NY, 0)
                    iy_s += ifelse(inside_down, NY, 0)
                    for ix = 1:NX
                        ix_s = ix + shift1
                        inside_up = ix_s > NX; inside_down = ix_s < 1
                        factor_x = ifelse(inside_up || inside_down, Int(bc[1]), 1)
                        ix_s += ifelse(inside_up, -NX, 0)
                        ix_s += ifelse(inside_down, NX, 0)

                        factor = factor_x * factor_y * factor_z * factor_t

                        # Read shifted fermion
                        x1 = factor * sf[1, ix_s, iy_s, iz_s, it_s, ig]
                        x2 = factor * sf[2, ix_s, iy_s, iz_s, it_s, ig]
                        x3 = factor * sf[3, ix_s, iy_s, iz_s, it_s, ig]

                        # Read gauge at shifted index (no shift_U needed)
                        # Compute U†*x: (U†)[i,j] = conj(U[j,i])
                        fsh[1, ix, iy, iz, it, ig] =
                            conj(Ug[1,1,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,1,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,1,ix_s,iy_s,iz_s,it_s])*x3
                        fsh[2, ix, iy, iz, it, ig] =
                            conj(Ug[1,2,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,2,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,2,ix_s,iy_s,iz_s,it_s])*x3
                        fsh[3, ix, iy, iz, it, ig] =
                            conj(Ug[1,3,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,3,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,3,ix_s,iy_s,iz_s,it_s])*x3
                    end
                end
            end
        end
    end
    # fshifted → f 복사
    copyto!(sf, fsh)
    return nothing
end

# Backward step: copyx[site] = U_μ(site) * copyx_old(site+μ̂)
# 게이지는 shift 불필요 (현재 site에서 읽음)
function _hop_step_backward!(copyx::WF3x1CPU, U_mu, mu::Int)
    # fermion만 shift: copyx.f → copyx.fshifted
    shift = ntuple(i -> (i == mu) ? 1 : 0, 4)
    _shifted_fermion_inplace!(copyx, shift)

    # fshifted에서 읽고 U*fshifted 결과를 f에 씀
    fsh = copyx.fshifted
    Ug = U_mu.U
    df = copyx.f
    NX = copyx.NX; NY = copyx.NY; NZ = copyx.NZ; NT = copyx.NT

    @inbounds for ig = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        x1 = fsh[1, ix, iy, iz, it, ig]
                        x2 = fsh[2, ix, iy, iz, it, ig]
                        x3 = fsh[3, ix, iy, iz, it, ig]

                        df[1, ix, iy, iz, it, ig] =
                            Ug[1,1,ix,iy,iz,it]*x1 + Ug[1,2,ix,iy,iz,it]*x2 + Ug[1,3,ix,iy,iz,it]*x3
                        df[2, ix, iy, iz, it, ig] =
                            Ug[2,1,ix,iy,iz,it]*x1 + Ug[2,2,ix,iy,iz,it]*x2 + Ug[2,3,ix,iy,iz,it]*x3
                        df[3, ix, iy, iz, it, ig] =
                            Ug[3,1,ix,iy,iz,it]*x1 + Ug[3,2,ix,iy,iz,it]*x2 + Ug[3,3,ix,iy,iz,it]*x3
                    end
                end
            end
        end
    end
    return nothing
end


# Preallocated 버퍼를 받는 버전 - 메모리 할당 완전 제거
function (hop::Hopping_operator)(x::WF3x1CPU, V::AbstractVector{T}) where {T<:WF3x1CPU}
    @assert length(V) == length(hop.P.various) "V의 길이는 paths의 개수와 같아야 합니다."
    
    U = hop.Gauge

    @inbounds for k in eachindex(V)
        copyx = V[k]
        
        # copyx를 x로 초기화 (inplace copy)
        substitute_fermion!(copyx, x)

        for step in hop.P.various[k].steps
            a = step.f
            if a > 0
                _hop_step_forward!(copyx, U[a], a)

            elseif a < 0
                _hop_step_backward!(copyx, U[-a], -a)
            
            else
                # a == 0: copyx를 x로 재설정
                substitute_fermion!(copyx, x)
            end
        end
    end

    return V
end







function PTC(W::Vector{Spin_matrix},hop::Hopping_operator,x::WF3x1) 

    result=similar(x)



    fields=hop(x)

    if length(W)==1
        for i in 1:length(fields)
            result+=W[1]*fields[i]
        end
    end


        


    if length(W)==length(fields)

        for i in 1:length(fields)
            result+=W[i]*fields[i]
        end
    end

    
    return result
    

end    


function Base.:length(hop::Hopping_operator)
    return length(hop.P)

end


function CPTC(hop::Hopping_operator,W::Vector{Spin_matrix},f::WF3x1)
    
    s=length(hop)
    cW=Clifford(W)
    result=similar(f)
    if length(cW)==s
        for i in 1:s
            f+=cW[i]*hop(f)[i]
        end
    end

    if length(cW)==1
        for i in 1:s
            f+=cW[1]*hop(f)[i]
        end
    end

    return result
end


function mean(x::WF3x1)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    NG = x.NG

    sum_x = 0.0

    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            sum_x += x[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end

    return sum_x / (NX * NY * NZ * NT * NC * NG)


end 

function Base.:+(x::WF3x1,a::Number)

    r=deepcopy(x)

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
                            r[ic, ix, iy, iz, it, ig] =x[ic,ix,iy,iz,it,ig] + a
                        end
                    end
                end
            end
        end
    end

    return r
end 

function Base.:-(x::WF3x1,a::Number)

    r=deepcopy(x)

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
                            r[ic, ix, iy, iz, it, ig] =x[ic,ix,iy,iz,it,ig] - a
                        end
                    end
                end
            end
        end
    end

    return r
end

function Base.:-(a::Number, x::WF3x1)
    r = deepcopy(x)
    for ic = 1:x.NC, it = 1:x.NT, iz = 1:x.NZ, iy = 1:x.NY, ix = 1:x.NX, ig = 1:x.NG
        r[ic, ix, iy, iz, it, ig] = a - x[ic, ix, iy, iz, it, ig]
    end
    return r
end



function Base.:minmax(x::WF3x1)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG

    first_val = x[1, 1, 1, 1, 1, 1]
    min_val = abs(first_val)
    max_val = abs(first_val)

    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            val = abs(x[ic, ix, iy, iz, it, ig])
                            if abs(val) < abs(min_val)
                                min_val = abs(val)
                            end
                            if abs(val) > abs(max_val)
                                max_val = abs(val)
                            end
                        end
                    end
                end
            end
        end
    end

    return min_val, max_val
end

function standardization(x::WF3x1)
    mu = mean(x)
    x = x - mu

    real = (x + conj(x)) / 2
    imag = (x - conj(x)) / 2im

    N = x.NX * x.NY * x.NZ * x.NT * x.NC * x.NG
    σr = sqrt(norm2(real) / N)
    σi = sqrt(norm2(imag) / N)

    x = real / σr + im * imag / σi

    return x
end




using Random

function g2vector_generator!(x::WF3x1; p::Real=0.5, rng::AbstractRNG=Random.default_rng())
    @assert 0 ≤ p ≤ 1 "p must be in [0,1]"
    NX, NY, NZ, NT, NC, NG = x.NX, x.NY, x.NZ, x.NT, x.NC, x.NG
    @inbounds for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            x[ic, ix, iy, iz, it, ig] = (rand(rng) < p) ? 1 : 0
                        end
                    end
                end
            end
        end
    end
    return x
end

function rand01_like(x::WF3x1; p::Real=0.5, rng::AbstractRNG=Random.default_rng())
    r = deepcopy(x)
    fill01!(r; p=p, rng=rng)
    return r
end


function spincolortrace(x::WF3x1,y::WF3x1)
    @assert size(x) == size(y) "Input vectors must have the same dimensions"
    NX, NY, NZ, NT, NC, NG = x.NX, x.NY, x.NZ, x.NT, x.NC, x.NG
    L=NX * NY * NZ * NT
    trace_sum = Array{ComplexF64}(undef, NX, NY, NZ, NT)
 
        @inbounds    for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        local_trace = 0.0 + 0.0im
                        for ig = 1:NG
                            for ic = 1:NC
                                local_trace += x[ic, ix, iy, iz, it, ig] * y[ic, ix, iy, iz, it, ig]
                            end
                        end
                        trace_sum[ix, iy, iz, it] = local_trace
                    end
                end
            end
        end

    return trace_sum
end


function Base.:*(v::Array{ComplexF64}, x::WF3x1)

    r = similar(x)
    NX, NY, NZ, NT, NC, NG = x.NX, x.NY, x.NZ, x.NT, x.NC, x.NG
    @assert length(v) == NT*NZ*NY*NX "Length of vector v must match the spatial-temporal dimensions of x"
    @inbounds for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                             r[ic, ix, iy, iz, it, ig] += v[ix, iy, iz, it]*x[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return r
end 












"""
    blockize(x::WF3x1, r::Int)

페르미온 벡터 `x`를 시공간 방향마다 `r`개씩 묶어 블록화한다.
NX, NY, NZ, NT 각각이 `r`로 나누어떨어져야 한다.

반환값: `Vector{WF3x1}` — 길이 `(NX÷r)*(NY÷r)*(NZ÷r)*(NT÷r)`.
각 원소는 원래와 같은 격자 크기의 WF3x1이며,
해당 블록에 속하는 사이트만 원래 값을 갖고 나머지는 0.
블록 순서는 (bx, by, bz, bt)를 row-major로 flatten한 것.
"""
function blockize(x::WF3x1, r::Int)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG

    @assert NX % r == 0 "NX=$NX 는 r=$r 로 나누어떨어져야 합니다."
    @assert NY % r == 0 "NY=$NY 는 r=$r 로 나누어떨어져야 합니다."
    @assert NZ % r == 0 "NZ=$NZ 는 r=$r 로 나누어떨어져야 합니다."
    @assert NT % r == 0 "NT=$NT 는 r=$r 로 나누어떨어져야 합니다."

    NBX = NX ÷ r
    NBY = NY ÷ r
    NBZ = NZ ÷ r
    NBT = NT ÷ r

    blocks = Vector{typeof(x)}(undef, NBX * NBY * NBZ * NBT)

    idx = 0
    @inbounds for bt = 1:NBT
        for bz = 1:NBZ
            for by = 1:NBY
                for bx = 1:NBX
                    idx += 1
                    blk = similar(x)   # 0으로 초기화된 WF3x1
                    for lt = 1:r
                        it = (bt - 1) * r + lt
                        for lz = 1:r
                            iz = (bz - 1) * r + lz
                            for ly = 1:r
                                iy = (by - 1) * r + ly
                                for lx = 1:r
                                    ix = (bx - 1) * r + lx
                                    for ig = 1:NG
                                        @simd for ic = 1:NC
                                            blk[ic, ix, iy, iz, it, ig] =
                                                x[ic, ix, iy, iz, it, ig]
                                        end
                                    end
                                end
                            end
                        end
                    end
                    blocks[idx] = blk
                end
            end
        end
    end

    return blocks
end

"""
    unblockize(blocks::Vector{WF3x1})

`blockize`의 역연산. 블록 벡터들을 모두 더해 원래 페르미온 벡터를 복원한다.
각 블록은 서로 겹치지 않는 사이트를 갖고 있으므로 단순 합산으로 복원 가능.
"""
function unblockize(blocks::Vector{T}) where {T<:WF3x1}
    f = similar(blocks[1])
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    NC, NG = f.NC, f.NG

    for blk in blocks
        @inbounds for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            @simd for ic = 1:NC
                                f[ic, ix, iy, iz, it, ig] +=
                                    blk[ic, ix, iy, iz, it, ig]
                            end
                        end
                    end
                end
            end
        end
    end

    return f
end


end # module




