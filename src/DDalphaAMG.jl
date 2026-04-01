module DDalphaAMG

using ..elementry_base
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using CUDA
export DDalphaAMG_Setup, CoarseVector
export setup_DDalphaAMG, solve_DDalphaAMG, vcycle
export restrict_f2c, prolongate_c2f, coarse_apply
export mr_smooth!, sap_smooth!
export block_orthonormalize!


const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

# ==========================================
#  粗格子ベクトル (Coarse Grid Vector)
# ==========================================

"""
    CoarseVector

粗格子上のベクトル。
data[(bidx-1)*Nv + v] でブロック bidx のテストベクトル番号 v の成分にアクセス。
"""
mutable struct CoarseVector
    data::Vector{ComplexF64}
    Nv::Int
    Nblocks::Int
end

CoarseVector(Nv::Int, Nblocks::Int) =
    CoarseVector(zeros(ComplexF64, Nv * Nblocks), Nv, Nblocks)

Base.similar(cv::CoarseVector) =
    CoarseVector(zeros(ComplexF64, length(cv.data)), cv.Nv, cv.Nblocks)
Base.copy(cv::CoarseVector) =
    CoarseVector(copy(cv.data), cv.Nv, cv.Nblocks)
Base.length(cv::CoarseVector) = length(cv.data)

function zero!(cv::CoarseVector)
    fill!(cv.data, zero(ComplexF64))
    return cv
end

function coarse_dot(a::CoarseVector, b::CoarseVector)
    return dot(a.data, b.data)
end

function coarse_norm(a::CoarseVector)
    return sqrt(real(dot(a.data, a.data)))
end

function coarse_axpby!(α, x::CoarseVector, β, y::CoarseVector)
    @inbounds @simd for i in eachindex(y.data)
        y.data[i] = α * x.data[i] + β * y.data[i]
    end
    return y
end


# ==========================================
#  セットアップ構造体 (Setup Structure)
# ==========================================

"""
    DDalphaAMG_Setup

DD-αAMG のセットアップ情報を保持する構造体。
- test_vectors: ブロック直交正規化されたテストベクトル
- r: 各方向のブロックサイズ
- Nv: テストベクトル数
- NBX,NBY,NBZ,NBT: 粗格子の各方向ブロック数
"""
mutable struct DDalphaAMG_Setup{T<:WF3x1}
    test_vectors::Vector{T}
    r::Int
    Nv::Int
    NX::Int; NY::Int; NZ::Int; NT::Int
    NC::Int; NG::Int
    NBX::Int; NBY::Int; NBZ::Int; NBT::Int
    Nblocks::Int
end


# ==========================================
#  ブロックユーティリティ
# ==========================================

@inline function block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
    return (bt-1)*NBZ*NBY*NBX + (bz-1)*NBY*NBX + (by-1)*NBX + bx
end

"""
    block_project!(f, bx, by, bz, bt, r)

ブロック(bx,by,bz,bt)外の全サイトをゼロにする (in-place)。
"""
function block_project!(f::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    NC, NG = f.NC, f.NG

    ix_lo = (bx-1)*r + 1; ix_hi = bx*r
    iy_lo = (by-1)*r + 1; iy_hi = by*r
    iz_lo = (bz-1)*r + 1; iz_hi = bz*r
    it_lo = (bt-1)*r + 1; it_hi = bt*r

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    if !(ix_lo <= ix <= ix_hi && iy_lo <= iy <= iy_hi &&
                         iz_lo <= iz <= iz_hi && it_lo <= it <= it_hi)
                        for ig = 1:NG
                            @simd for ic = 1:NC
                                f[ic, ix, iy, iz, it, ig] = zero(ComplexF64)
                            end
                        end
                    end
                end
            end
        end
    end
    return f
end

"""
    block_copy!(dest, src, bx, by, bz, bt, r)

src のブロック(bx,by,bz,bt)内の値を dest の同じ位置にコピーする (in-place)。
dest のブロック外は変更しない。
"""
function block_copy!(dest::WF3x1, src::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = src.NC, src.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            dest[ic, ix, iy, iz, it, ig] = src[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return dest
end

"""
    block_zero!(f, bx, by, bz, bt, r)

f のブロック(bx,by,bz,bt)内の値をゼロにする (in-place)。
"""
function block_zero!(f::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = f.NC, f.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            f[ic, ix, iy, iz, it, ig] = zero(ComplexF64)
                        end
                    end
                end
            end
        end
    end
    return f
end

"""
    block_axpby!(α, x, β, y, bx, by, bz, bt, r)

y[block] = α * x[block] + β * y[block] をブロック内でのみ実行 (in-place)。
"""
function block_axpby!(α, x::WF3x1, β, y::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = x.NC, x.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            y[ic, ix, iy, iz, it, ig] =
                                α * x[ic, ix, iy, iz, it, ig] + β * y[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return y
end

"""
    block_inner_product(x, y, bx, by, bz, bt, r) → ComplexF64

ブロック(bx,by,bz,bt)内のみの内積 ⟨x|y⟩_B。
"""
function block_inner_product(x::WF3x1, y::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = x.NC, x.NG
    val = zero(ComplexF64)
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            val += conj(x[ic, ix, iy, iz, it, ig]) * y[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return val
end


# ==========================================
#  ブロック直交正規化 (Block Orthonormalization)
# ==========================================

"""
    block_orthonormalize!(vectors, r)

各ブロック内でテストベクトルを Modified Gram-Schmidt で直交正規化する。
処理後: 各ブロックBについて ⟨ψᵢ|ψⱼ⟩_B = δᵢⱼ
"""
function block_orthonormalize!(vectors::Vector{T}, r::Int) where {T<:WF3x1}
    Nv = length(vectors)
    v1 = vectors[1]
    NX, NY, NZ, NT = v1.NX, v1.NY, v1.NZ, v1.NT
    NC, NG = v1.NC, v1.NG
    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    block_dof = r^4 * NC * NG

    # ローカルバッファを事前確保 (ブロック毎に再利用)
    local_vecs = [Vector{ComplexF64}(undef, block_dof) for _ in 1:Nv]

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        # ブロック内データを抽出
        for v = 1:Nv
            idx = 0
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        idx += 1
                        local_vecs[v][idx] = vectors[v][ic, ix, iy, iz, it, ig]
                    end
                end
            end
        end

        # Modified Gram-Schmidt
        for v = 1:Nv
            for u = 1:(v-1)
                proj = dot(local_vecs[u], local_vecs[v])
                local_vecs[v] .-= proj .* local_vecs[u]
            end
            nrm = norm(local_vecs[v])
            if nrm > 1e-14
                local_vecs[v] ./= nrm
            end
        end

        # 書き戻し
        for v = 1:Nv
            idx = 0
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        idx += 1
                        vectors[v][ic, ix, iy, iz, it, ig] = local_vecs[v][idx]
                    end
                end
            end
        end
    end

    return vectors
end


# ==========================================
#  制限演算子 (Restriction: fine → coarse)
# ==========================================

"""
    restrict_f2c(setup, x) → CoarseVector

細格子ベクトルを粗格子に射影:
  v_coarse[B, i] = ⟨ψᵢ|x⟩_B  (ブロックB内の内積)
"""
function restrict_f2c(setup::DDalphaAMG_Setup, x::WF3x1)
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    cv = CoarseVector(Nv, setup.Nblocks)

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        for v = 1:Nv
            val = zero(ComplexF64)
            tv = setup.test_vectors[v]
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        val += conj(tv[ic, ix, iy, iz, it, ig]) * x[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            cv.data[(bidx-1)*Nv + v] = val
        end
    end

    return cv
end


# ==========================================
#  延長演算子 (Prolongation: coarse → fine)
# ==========================================

"""
    prolongate_c2f(setup, cv) → WF3x1

粗格子ベクトルを細格子に補間:
  x(site) = Σᵢ c[B(site), i] * ψᵢ(site)
"""
function prolongate_c2f(setup::DDalphaAMG_Setup, cv::CoarseVector)
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    f = similar(setup.test_vectors[1])

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        base = (bidx-1)*Nv
        for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
            it = (bt-1)*r + lt; iz = (bz-1)*r + lz
            iy = (by-1)*r + ly; ix = (bx-1)*r + lx
            for ig = 1:NG
                for ic = 1:NC
                    val = zero(ComplexF64)
                    for v = 1:Nv
                        val += cv.data[base + v] * setup.test_vectors[v][ic, ix, iy, iz, it, ig]
                    end
                    f[ic, ix, iy, iz, it, ig] = val
                end
            end
        end
    end

    return f
end


# ==========================================
#  粗格子演算子 (Coarse Grid Operator, matrix-free)
# ==========================================

"""
    coarse_apply(setup, D, cv) → CoarseVector

粗格子Dirac演算子の適用: Dc = R · D · P
  (Prolongate → Apply D on fine grid → Restrict)
"""
function coarse_apply(setup::DDalphaAMG_Setup, D, cv::CoarseVector)
    fine_v = prolongate_c2f(setup, cv)
    Dfine_v = D * fine_v
    return restrict_f2c(setup, Dfine_v)
end


# ==========================================
#  ランダムフェルミオン生成 (Setup用)
# ==========================================

function random_fermion!(x::WF3x1)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG
    @inbounds for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX
        for ig = 1:NG
            for ic = 1:NC
                x[ic, ix, iy, iz, it, ig] = randn() + im * randn()
            end
        end
    end
    return x
end


# ==========================================
#  スムーザー (Smoothers)
# ==========================================

"""
    mr_smooth!(x, D, b; niter)

Minimal Residual (MR) スムーザー。
各イテレーション:
  r = b - D*x,  α = ⟨D*r, r⟩ / ⟨D*r, D*r⟩,  x += α*r
"""
function mr_smooth!(x::WF3x1, D, b::WF3x1; niter::Int=4)
    for _ = 1:niter
        r = b - D * x
        Dr = D * r
        num = inner_product(Dr, r)
        den = inner_product(Dr, Dr)
        if abs(den) < 1e-30
            break
        end
        α = num / den
        axpby!(α, r, 1, x)
    end
    return x
end


"""
    sap_smooth!(x, D, b, setup; niter, nblock_mr)

Schwarz Alternating Procedure (SAP) スムーザー。
ブロック毎にローカル Minimal Residual ステップを実行 (multiplicative Schwarz)。
各ブロック内で Dirichlet B.C. を用い、D_BB * δ ≈ r_B を近似的に解く。

バッファを事前確保し、ブロックループ内でのメモリ割り当てを回避。
内積はブロックローカルで計算し、フル格子演算を最小化。
"""
function sap_smooth!(x::WF3x1, D, b::WF3x1, setup::DDalphaAMG_Setup;
                     niter::Int=1, nblock_mr::Int=4)
    r_blk = setup.r
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT

    # バッファを事前確保（ループ内で再利用）
    δ      = similar(x)
    local_r = similar(x)
    Alr    = similar(x)
    res    = similar(x)

    for _ = 1:niter
        # グローバル残差: res = b - D*x
        mul!(res, D, x)           # res = D*x
        axpby!(1, b, -1, res)     # res = b - D*x  → axpby!(1, b, -1, res) => res = 1*b + (-1)*res

        for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
            # δ, local_r のブロック内をゼロ/コピーで初期化
            # local_r は全体ゼロにしてからブロック内だけコピー（D適用時にブロック外がゼロであることを保証）
            block_zero!(δ, bx, by, bz, bt, r_blk)
            block_project!(local_r, bx, by, bz, bt, r_blk)  # ブロック外をゼロに
            block_copy!(local_r, res, bx, by, bz, bt, r_blk) # ブロック内に res_B をコピー

            for _ = 1:nblock_mr
                # Alr = project_B(D * local_r)  (local_r はブロック内のみ非ゼロ)
                mul!(Alr, D, local_r)
                block_project!(Alr, bx, by, bz, bt, r_blk)

                num = block_inner_product(Alr, local_r, bx, by, bz, bt, r_blk)
                den = block_inner_product(Alr, Alr, bx, by, bz, bt, r_blk)
                if abs(den) < 1e-30
                    break
                end
                α = num / den

                # δ_B += α * local_r_B,  local_r_B -= α * Alr_B
                block_axpby!(α, local_r, 1, δ, bx, by, bz, bt, r_blk)
                block_axpby!(-α, Alr, 1, local_r, bx, by, bz, bt, r_blk)
            end

            # グローバル解の更新: x_B += δ_B
            block_axpby!(1, δ, 1, x, bx, by, bz, bt, r_blk)

            # Multiplicative Schwarz: res -= D*δ（後続ブロックに反映）
            # δ のブロック外はゼロなので D*δ は正しい
            block_project!(δ, bx, by, bz, bt, r_blk)
            mul!(Alr, D, δ)  # Alr を一時バッファとして再利用
            axpby!(-1, Alr, 1, res)
        end
    end
    return x
end


# ==========================================
#  粗格子 GMRES (Coarse Grid Solver)
# ==========================================

"""
    coarse_gmres(setup, D, b_c; maxiter, restart, tol) → CoarseVector

粗格子上の GMRES ソルバー。Dc * x_c = b_c を解く。
Dc は matrix-free で適用: Dc(v) = R · D · P(v)
"""
function coarse_gmres(setup::DDalphaAMG_Setup, D, b_c::CoarseVector;
                      maxiter::Int=50, restart::Int=20, tol::Float64=1e-1)
    x_c = similar(b_c)

    for _ = 1:maxiter
        # 残差: r = b - Dc*x
        r_c = copy(b_c)
        Dc_x = coarse_apply(setup, D, x_c)
        coarse_axpby!(-1.0, Dc_x, 1.0, r_c)

        β = coarse_norm(r_c)
        if β < tol
            return x_c
        end

        # Arnoldi 過程
        m = min(restart, length(b_c.data))
        V = [similar(b_c) for _ in 1:m+1]
        V[1].data .= r_c.data ./ β
        H = zeros(ComplexF64, m+1, m)

        j_final = m
        for j = 1:m
            w = coarse_apply(setup, D, V[j])

            # Modified Gram-Schmidt
            for i = 1:j
                H[i,j] = coarse_dot(V[i], w)
                coarse_axpby!(-H[i,j], V[i], 1.0, w)
            end
            H[j+1,j] = coarse_norm(w)

            if abs(H[j+1,j]) < 1e-14
                j_final = j
                break
            end
            V[j+1].data .= w.data ./ H[j+1,j]
        end

        # 最小二乗問題: min ||β*e₁ - H*y||
        e1 = zeros(ComplexF64, j_final + 1)
        e1[1] = β
        y = H[1:j_final+1, 1:j_final] \ e1

        for k = 1:j_final
            coarse_axpby!(y[k], V[k], 1.0, x_c)
        end

        # 収束確認
        check_r = copy(b_c)
        coarse_axpby!(-1.0, coarse_apply(setup, D, x_c), 1.0, check_r)
        if coarse_norm(check_r) < tol
            return x_c
        end
    end

    return x_c
end


# ==========================================
#  V-cycle
# ==========================================

"""
    vcycle(D, b, setup; pre_smooth, post_smooth, coarse_tol, smoother)

1回のマルチグリッド V-cycle を実行し、D*x ≈ b の近似解を返す。

1. Pre-smoothing (MR or SAP)
2. 残差を粗格子に射影: r_c = R(b - D*x)
3. 粗格子で GMRES: Dc * e_c = r_c
4. 補正を細格子に戻す: x += P(e_c)
5. Post-smoothing
"""
function vcycle(D, b::WF3x1, setup::DDalphaAMG_Setup;
                pre_smooth::Int=4, post_smooth::Int=4,
                coarse_maxiter::Int=50, coarse_tol::Float64=1e-1,
                smoother::Symbol=:mr)
    x = similar(b)

    # 1. Pre-smoothing
    if smoother == :sap
        sap_smooth!(x, D, b, setup; niter=1, nblock_mr=pre_smooth)
    else
        mr_smooth!(x, D, b; niter=pre_smooth)
    end

    # 2. 残差計算 → 粗格子に射影
    r_fine = b - D * x
    r_coarse = restrict_f2c(setup, r_fine)

    # 3. 粗格子求解
    e_coarse = coarse_gmres(setup, D, r_coarse;
                            maxiter=coarse_maxiter, tol=coarse_tol)

    # 4. 細格子への補正
    correction = prolongate_c2f(setup, e_coarse)
    axpby!(1, correction, 1, x)

    # 5. Post-smoothing
    if smoother == :sap
        sap_smooth!(x, D, b, setup; niter=1, nblock_mr=post_smooth)
    else
        mr_smooth!(x, D, b; niter=post_smooth)
    end

    return x
end


# ==========================================
#  FGMRES (Flexible GMRES) 外部ソルバー
# ==========================================

"""
    fgmres(D, b, setup; x0, maxiter, restart, tol, verbose, smoother, ...)

Flexible GMRES (V-cycle プリコンディショナー付き)。
D*x = b を解く。

FGMRES は各イテレーションで異なるプリコンディショナーを許容するため、
非線形マルチグリッド V-cycle を変動プリコンディショナーとして使用可能。
"""
function fgmres(D, b::WF3x1, setup::DDalphaAMG_Setup;
                x0::Union{Nothing,WF3x1}=nothing,
                maxiter::Int=100, restart::Int=20, tol::Float64=1e-10,
                verbose::Bool=true, smoother::Symbol=:mr,
                pre_smooth::Int=4, post_smooth::Int=4,
                coarse_tol::Float64=1e-1)

    x = x0 === nothing ? similar(b) : deepcopy(x0)
    b_norm = sqrt(real(inner_product(b, b)))
    if b_norm < 1e-30
        return x
    end

    for cycle = 1:maxiter
        r = b - D * x
        β = sqrt(real(inner_product(r, r)))

        if verbose
            println("FGMRES cycle $cycle: relative residual = $(β/b_norm)")
        end

        if β / b_norm < tol
            if verbose
                println("FGMRES converged! ($(cycle) cycles)")
            end
            return x
        end

        # Arnoldi with flexible preconditioning + Givens rotation for early stop
        m = restart
        V = Vector{typeof(b)}(undef, m + 1)
        Z = Vector{typeof(b)}(undef, m)
        H = zeros(ComplexF64, m + 1, m)

        # Givens回転用
        cs = zeros(ComplexF64, m)
        sn = zeros(ComplexF64, m)
        g  = zeros(ComplexF64, m + 1)
        g[1] = β

        V[1] = r / β

        j_final = m
        converged = false
        for j = 1:m
            # V-cycle プリコンディショナー: M⁻¹ * v_j
            Z[j] = vcycle(D, V[j], setup;
                          pre_smooth=pre_smooth, post_smooth=post_smooth,
                          coarse_tol=coarse_tol, smoother=smoother)

            w = D * Z[j]

            # Modified Gram-Schmidt
            for i = 1:j
                H[i,j] = inner_product(V[i], w)
                axpby!(-H[i,j], V[i], 1, w)
            end
            H[j+1,j] = sqrt(real(inner_product(w, w)))

            if abs(H[j+1,j]) < 1e-14
                j_final = j
                # 既存の Givens 回転を適用
                for i = 1:(j-1)
                    tmp = cs[i] * H[i,j] + sn[i] * H[i+1,j]
                    H[i+1,j] = -conj(sn[i]) * H[i,j] + conj(cs[i]) * H[i+1,j]
                    H[i,j] = tmp
                end
                converged = true
                break
            end
            V[j+1] = w / H[j+1,j]

            # 既存の Givens 回転を H[:,j] に適用
            for i = 1:(j-1)
                tmp = cs[i] * H[i,j] + sn[i] * H[i+1,j]
                H[i+1,j] = -conj(sn[i]) * H[i,j] + conj(cs[i]) * H[i+1,j]
                H[i,j] = tmp
            end

            # 新しい Givens 回転を計算
            denom = sqrt(abs(H[j,j])^2 + abs(H[j+1,j])^2)
            cs[j] = conj(H[j,j]) / denom
            sn[j] = H[j+1,j] / denom
            H[j,j] = denom
            H[j+1,j] = zero(ComplexF64)

            # 残差ノルム更新
            g[j+1] = -conj(sn[j]) * g[j]
            g[j]   = cs[j] * g[j]

            res_norm = abs(g[j+1]) / b_norm
            if verbose
                println(" maxiter = $cycle, restart = $j, relative residual = $res_norm")
            end

            if res_norm < tol
                j_final = j
                converged = true
                break
            end
        end

        # 後退代入で y を求解
        y = zeros(ComplexF64, j_final)
        for i = j_final:-1:1
            y[i] = g[i]
            for k = (i+1):j_final
                y[i] -= H[i,k] * y[k]
            end
            y[i] /= H[i,i]
        end

        # 解の更新: x += Z * y
        for k = 1:j_final
            axpby!(y[k], Z[k], 1, x)
        end

        if converged
            if verbose
                println("FGMRES converged at cycle $cycle, step $(j_final)!")
            end
            return x
        end
    end

    if verbose
        r_final = b - D * x
        println("FGMRES: max iterations reached. " *
                "Final relative residual = $(sqrt(real(inner_product(r_final, r_final)))/b_norm)")
    end
    return x
end


# ==========================================
#  セットアップ (Full Setup)
# ==========================================

"""
    setup_DDalphaAMG(D, template, r, Nv; nsmooth, verbose) → DDalphaAMG_Setup

DD-αAMG の完全セットアップを行う。

手順:
1. ランダムテストベクトルを Nv 個生成
2. MR スムージングで低モード成分を強化 (D*ψ ≈ 0 の近似求解)
3. ブロック直交正規化

引数:
- D: Dirac 演算子 (Wilson_Dirac_operator 等)
- template: 格子サイズ参照用の WF3x1
- r: 各方向のブロックサイズ (NX,NY,NZ,NT が r で割り切れること)
- Nv: テストベクトル数 (推奨: 20〜30)
- nsmooth: セットアップスムージングの反復数 (デフォルト: 8)

使用例:
```julia
setup = setup_DDalphaAMG(D, x, 2, 24; nsmooth=16)
x = solve_DDalphaAMG(D, b, setup; tol=1e-10)
```
"""
function setup_DDalphaAMG(D, template::WF3x1, r::Int, Nv::Int;
                          nsmooth::Int=8, verbose::Bool=true, coarse_vectors::Union{Nothing,Vector{WF3x1}}=nothing)
    NX, NY, NZ, NT = template.NX, template.NY, template.NZ, template.NT
    NC, NG = template.NC, template.NG

    @assert NX % r == 0 "NX=$NX は r=$r で割り切れる必要があります"
    @assert NY % r == 0 "NY=$NY は r=$r で割り切れる必要があります"
    @assert NZ % r == 0 "NZ=$NZ は r=$r で割り切れる必要があります"
    @assert NT % r == 0 "NT=$NT は r=$r で割り切れる必要があります"

    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    Nblocks = NBX * NBY * NBZ * NBT

    if verbose
        println("DD-αAMG Setup:")
        println("  格子サイズ: $(NX)×$(NY)×$(NZ)×$(NT)")
        println("  ブロックサイズ: $(r)^4")
        println("  粗格子: $(NBX)×$(NBY)×$(NBZ)×$(NBT) = $(Nblocks) blocks")
        println("  テストベクトル数: $Nv")
        println("  粗格子自由度: $(Nblocks * Nv)")
    end

    # 1. ランダムテストベクトル生成
    if coarse_vectors !== nothing
        @assert length(coarse_vectors) == Nv "coarse_vectors の長さは Nv と一致する必要があります"
        test_vectors = coarse_vectors
        if verbose
            println("  外部から提供されたテストベクトルを使用")
        end
    else
        # 2. スムージング: MR で D*ψ ≈ 0 を近似求解 → 低モード成分抽出
        test_vectors = [similar(template) for _ in 1:Nv]
        for v in 1:Nv
            random_fermion!(test_vectors[v])
        end
        if verbose
            println("  テストベクトルをスムージング中 ($nsmooth MR iterations)...")
        end
        zero_rhs = similar(template)  # ゼロ右辺
        for v in 1:Nv
            mr_smooth!(test_vectors[v], D, zero_rhs; niter=nsmooth)
        end
    end



    # 3. ブロック直交正規化
    if verbose
        println("  ブロック直交正規化中...")
    end
    block_orthonormalize!(test_vectors, r)

    setup = DDalphaAMG_Setup{typeof(template)}(
        test_vectors, r, Nv, NX, NY, NZ, NT, NC, NG,
        NBX, NBY, NBZ, NBT, Nblocks
    )

    if verbose
        println("  セットアップ完了!")
    end

    return setup
end


# ==========================================
#  完全ソルバー (Full Solve)
# ==========================================

"""
    solve_DDalphaAMG(D, b, setup; x0, tol, maxiter, restart, verbose, smoother, ...)

DD-αAMG を用いて D*x = b を解く。
FGMRES 外部ソルバー + V-cycle プリコンディショナーを使用。

引数:
- D: Dirac 演算子
- b: 右辺ベクトル (WF3x1)
- setup: DDalphaAMG_Setup (setup_DDalphaAMG で生成)
- tol: 収束許容誤差 (デフォルト: 1e-10)
- smoother: :mr (Minimal Residual, デフォルト) または :sap (Schwarz Alternating Procedure)
- pre_smooth, post_smooth: スムージング反復数 (デフォルト: 各4)

使用例:
```julia
setup = setup_DDalphaAMG(D, x, 2, 24)
solution = solve_DDalphaAMG(D, b, setup; tol=1e-10, smoother=:mr)
residual = norm2(D * solution - b)
```
"""
function solve_DDalphaAMG(D, b::WF3x1, setup::DDalphaAMG_Setup;
                          x0::Union{Nothing,WF3x1}=nothing,
                          tol::Float64=1e-10, maxiter::Int=100, restart::Int=20,
                          verbose::Bool=true, smoother::Symbol=:mr,
                          pre_smooth::Int=4, post_smooth::Int=4)
    return fgmres(D, b, setup;
                  x0=x0, tol=tol, maxiter=maxiter, restart=restart,
                  verbose=verbose, smoother=smoother,
                  pre_smooth=pre_smooth, post_smooth=post_smooth)
end


end # module DDalphaAMG
