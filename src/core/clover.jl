# Clover field strength and the clover (Sheikholeslami-Wohlert) Dirac
# operator.

# ============================================================
# Clover Dirac operator  (Sheikholeslami-Wohlert)
# D_cl = D_Wilson + (i*c_SW*κ) Σ_{μ<ν} σ_μν F̂_μν
# Here σ_μν = (i/2)[γ_μ,γ_ν] and F̂_μν = (Q_μν - Q_μν†)/8.
# ============================================================


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
반에르미트 (anti-hermitian) 결과를 반환.

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
