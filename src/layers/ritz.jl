# Ritz_Layer: the Ritz-value loss. It projects the Dirac operator onto the span
# of the input fields and returns the Rayleigh-Ritz values.

mutable struct Ritz_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    # --- 캐시 버퍼 (Dirac 재계산 방지) ---
    _Dz::Vector{Any}        # forward에서 D*z 캐시 → DeltaRule!에서 재사용
    _DdagDz::Vector{Any}    # D†Dz 캐시 (DeltaRule!에서 Dirac ops 절반)
    _VD::Vector{Any}         # DeltaRule! VD 버퍼 (CPU path only)
    _work::Vector{Any}       # DeltaRule! workspace [ψ, r] (CPU path only)
    _ritz_init::Bool
    # Gram matrix + eigen 캐시 — CPU path
    _A::Matrix{ComplexF64}
    _B::Matrix{ComplexF64}
    _C::Matrix{ComplexF64}
    _Bfac::Matrix{ComplexF64}
    _evals::Vector{Float64}
    _evecs::Matrix{ComplexF64}
    _m::Vector{ComplexF64}
    function Ritz_Layer(D::Any, activation::ActivationFunction= Identity(),Out =1)
        new(D, activation, Out,
            Any[], Any[], Any[], Any[], false,
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Float64[], Matrix{ComplexF64}(undef,0,0), ComplexF64[])
    end
end 

# --- CuArray 표시 시 CUDA context 에러 방지 ---
function Base.show(io::IO, layer::Ritz_Layer)
    print(io, "Ritz_Layer(Out=$(layer.Out))")
end

function _get_blocks(layer::Ritz_Layer)
    try
        U = layer.D.U
        if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
            return U[1].blockinfo.blocks
        end
    catch; end
    return nothing
end

function _has_gpu(layer::Ritz_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
end

# --- Ritz_Layer ---
function _to_cpu(layer::Ritz_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    # Clear GPU/CPU cached workspaces after moving D.
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
    return layer
end

function _shared_layer_gauge(layer::Ritz_Layer)
    try
        return layer.D.U
    catch
        return nothing
    end
end

# --- Ritz_Layer ---
function _to_gpu(layer::Ritz_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
    # Ensure Ritz layer internals follow the selected GPU representation.
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
    return layer
end

function _to_gpu(layer::Ritz_Layer, blocks, shared_U)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks, shared_U=shared_U)
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
    return layer
end

_refresh_layer!(layer::Ritz_Layer, D_new, U_new) = (layer.D = D_new)

# function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})

#     l=length(x)
#     # V=x
#     A= Matrix{ComplexF64}(undef, l, l)
#     B= Matrix{ComplexF64}(undef, l, l)
#     D=layer.D
# # DDAGD LatticeDiracOperators


#     for i in 1:l
#         for j in 1:l
#             y=similar(x[j])
#             mul!(y,D,x[j])
#             u=similar(y)
#             mul!(u,D,x[i])
#             A[i,j]=inner_product(u,y)
#             B[i,j]=inner_product(x[i],x[j])
#         end
#     end
#     # println(A)
#     # println(B)

#     evals, evecs = eigen(A, B)

#     lambda=0
#     for i in 1:length(evals)
#         lambda+=real(evals[i])
#     end

#     return lambda
#     # return A,B
# end

function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)

    # 1) Compute Dx[i] = D * x[i] and DdagDx[i] = D' * D * x[i].
    need_reinit = !layer._ritz_init || length(layer._Dz) != l
    if !need_reinit
        # Reinitialize when CPU/GPU field type changes.
        need_reinit = typeof(layer._Dz[1]) != typeof(x[1])
    end
    if need_reinit
        layer._Dz = Vector{Any}(undef, l)
        layer._DdagDz = Vector{Any}(undef, l)
        for i in 1:l
            layer._Dz[i] = _alloc_result_wf(x[1])
            layer._DdagDz[i] = _alloc_result_wf(x[1])
        end
        layer._ritz_init = true
    end

    Dx = layer._Dz
    DdagDx = layer._DdagDz

    # GPU fused path: zero-alloc Wilson kernel
    _is_gpu_ritz = x[1] isa WF3x1GPU
    if _is_gpu_ritz
        _kappa = _dirac_kappa(layer.D)
        _U = layer.D.U
        @inbounds for i in 1:l
            fused_wilson_mul_gpu!(Dx[i], x[i], _U, _kappa, false)       # Dx = D * x
            fused_wilson_mul_gpu!(DdagDx[i], Dx[i], _U, _kappa, true)   # DdagDx = D' * D * x
        end
    else
        @inbounds for i in 1:l
            mul!(Dx[i], layer.D, x[i])       # Dx = D * x
            mul!(DdagDx[i], layer.D', Dx[i]) # DdagDx = D' * D * x
        end
    end

    # 2) Gram matrices: A_ij=<Dx_i, Dx_j>, B_ij=<x_i, x_j>
    # CPU path buffers (always kept for CPU fallback)
    if size(layer._A) != (l, l)
        layer._A = Matrix{ComplexF64}(undef, l, l)
        layer._B = Matrix{ComplexF64}(undef, l, l)
        layer._C = Matrix{ComplexF64}(undef, l, l)
        layer._Bfac = Matrix{ComplexF64}(undef, l, l)
        layer._evals = Vector{Float64}(undef, l)
        layer._evecs = Matrix{ComplexF64}(undef, l, l)
        layer._m = Vector{ComplexF64}(undef, l)
    end
    A = layer._A
    B = layer._B

    Agpu = nothing
    Bgpu = nothing
    if _is_gpu_ritz
        Agpu = get_shared_matrix!(:ritz_A, l)
        Bgpu = get_shared_matrix!(:ritz_B, l)
        ritz_gram_matrices_gpu!(Agpu, Bgpu, x, Dx)
        # Keep A and B on GPU for the generalized eigensolve.
    else
        @inbounds for i in 1:l
            bii = inner_product(x[i],  x[i])
            aii = inner_product(Dx[i], Dx[i])
            B[i,i] = bii
            A[i,i] = aii

            for j in i+1:l
                bij = inner_product(x[i],  x[j])
                aij = inner_product(Dx[i], Dx[j])

                B[i,j] = bij
                B[j,i] = conj(bij)

                A[i,j] = aij
                A[j,i] = conj(aij)
            end
        end
    end

    # 3) NaN check
    if _is_gpu_ritz
        if any(isnan, Agpu) || any(isinf, Agpu) || any(isnan, Bgpu) || any(isinf, Bgpu)
            n_nanA = count(isnan, Agpu); n_infA = count(isinf, Agpu)
            n_nanB = count(isnan, Bgpu); n_infB = count(isinf, Bgpu)
            println("[Ritz NaN] A: $(n_nanA) NaN, $(n_infA) Inf / $(length(Agpu)) | B: $(n_nanB) NaN, $(n_infB) Inf / $(length(Bgpu))")
            for i in 1:l
                _xf = hasproperty(x[i], :f) ? x[i].f : nothing
                _dxf = hasproperty(Dx[i], :f) ? Dx[i].f : nothing
                if _xf !== nothing && (any(isnan, _xf) || any(isinf, _xf))
                    println("[Ritz NaN] x[$i] contains NaN/Inf")
                end
                if _dxf !== nothing && (any(isnan, _dxf) || any(isinf, _dxf))
                    println("[Ritz NaN] Dx[$i] contains NaN/Inf")
                end
            end
            for (mu, ui) in enumerate(layer.D.U)
                if hasproperty(ui, :U) && ui.U isa CuArray
                    n_bad = count(x -> isnan(x) || isinf(x), ui.U)
                    n_bad > 0 && println("[Ritz NaN] D.U[$mu]: $n_bad / $(length(ui.U)) NaN/Inf")
                end
            end
        end
    else
        if any(isnan, A) || any(isinf, A) || any(isnan, B) || any(isinf, B)
            n_nanA = count(isnan, A); n_infA = count(isinf, A)
            n_nanB = count(isnan, B); n_infB = count(isinf, B)
            println("[Ritz NaN] A: $(n_nanA) NaN, $(n_infA) Inf / $(length(A)) | B: $(n_nanB) NaN, $(n_infB) Inf / $(length(B))")
            for i in 1:l
                _xf = hasproperty(x[i], :f) ? x[i].f : nothing
                _dxf = hasproperty(Dx[i], :f) ? Dx[i].f : nothing
                if _xf !== nothing && (_xf isa AbstractArray) && (any(isnan, _xf) || any(isinf, _xf))
                    println("[Ritz NaN] x[$i] contains NaN/Inf")
                end
                if _dxf !== nothing && (_dxf isa AbstractArray) && (any(isnan, _dxf) || any(isinf, _dxf))
                    println("[Ritz NaN] Dx[$i] contains NaN/Inf")
                end
            end
            for (mu, ui) in enumerate(layer.D.U)
                if hasproperty(ui, :U) && ui.U isa CuArray
                    n_bad = count(x -> isnan(x) || isinf(x), ui.U)
                    n_bad > 0 && println("[Ritz NaN] D.U[$mu]: $n_bad / $(length(ui.U)) NaN/Inf")
                end
            end
        end
    end

    # 4) Generalized eigenvalue decomposition + m_k
    if _is_gpu_ritz
        evals_out, evecs_out, _ = CUDA.CUSOLVER.hegvd!(1, 'V', 'U', Agpu, Bgpu)
        copyto!(layer._evals, evals_out)
        copyto!(layer._evecs, evecs_out)
        # CUSOLVER hegvd! returns B-orthonormal eigenvectors, so m[k] = 1.
        fill!(layer._m, one(ComplexF64))
    else
        C = layer._C
        Bfac = layer._Bfac
        copyto!(C, A)
        copyto!(Bfac, B)
        cholB = cholesky!(Hermitian(Bfac))
        ldiv!(adjoint(cholB.U), C)
        rdiv!(C, cholB.U)
        F = eigen!(Hermitian(C))
        copyto!(layer._evals, F.values)
        copyto!(layer._evecs, F.vectors)
        ldiv!(cholB.U, layer._evecs)
        # m_k = u_k' * B * u_k.
        m = layer._m
        @inbounds for k in 1:l
            s = zero(ComplexF64)
            for i in 1:l, j in 1:l
                s += conj(layer._evecs[i,k]) * B[i,j] * layer._evecs[j,k]
            end
            m[k] = s
        end
    end

    return real(sum(layer._evals))
end

function gradient(layer:: Ritz_Layer , ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

end 

function DeltaRule(layer::Ritz_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    l = length(z)

    # forward에서 캐싱된 결과 재사용
    evals  = layer._evals
    evecs  = layer._evecs
    m      = layer._m
    DdagDz = layer._DdagDz

    # result 버퍼 한 번만 할당
    result = Vector{WF3x1}(undef, l)
    for i in 1:l
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end
    
    # VD 버퍼도 루프 밖에서 미리 할당
    VD = Vector{WF3x1}(undef, l)
    for i in 1:l
        VD[i] = deepcopy(z[1])
    end
    
    ψ = deepcopy(z[1])
    r = deepcopy(z[1])
    
    # D†D ψ_k = Σ_j u_{jk} D†D z_j (Dirac ops 0회)
    for k in 1:l
        zero_wf!(ψ)
        zero_wf!(r)
        for j in 1:l
            axpby!(evecs[j,k], z[j], 1, ψ)
            axpby!(evecs[j,k], DdagDz[j], 1, r)
        end
        axpby!(1, r, -evals[k], ψ)
        conj_fermion!(VD[k], ψ)
    end

    for i in 1:l
        for k in 1:l
            axpby!(evecs[i,k]/m[k],VD[k], 1, result[i])
        end
    end

    return 2*result
end

function DeltaRule!(layer::Ritz_Layer, ::Identity, z::Vector{WF3x1}, delta::Vector{WF3x1}, batch::Int)

    l = length(z)
    D = layer.D
    for i in 1:length(delta)
        zero_wf!(delta[i])
    end

    # forward에서 캐싱된 eigen 결과 재사용 (Gram matrix + inner_product 6회 + eigen 재계산 제거)
    evals = layer._evals
    evecs = layer._evecs
    m     = layer._m
    DdagDz = layer._DdagDz   # D†D z[j] — forward에서 캐시됨

    # VD, ψ, r 버퍼 캐시 (매 epoch deepcopy 방지)
    need_vd_reinit = isempty(layer._VD) || length(layer._VD) != l
    if !need_vd_reinit
        need_vd_reinit = typeof(layer._VD[1]) != typeof(z[1])
    end
    if need_vd_reinit
        layer._VD = Vector{Any}(undef, l)
        for i in 1:l
            layer._VD[i] = _alloc_result_wf(z[1])
        end
        layer._work = Any[_alloc_result_wf(z[1]), _alloc_result_wf(z[1])]  # [ψ, r] only (y, f 불필요)
    end

    VD = layer._VD
    ψ  = layer._work[1]
    r  = layer._work[2]
    
    # D†D ψ_k = Σ_j u_{jk} D†D z_j (Dirac ops 0회 — DdagDz 캐시 재사용)
    for k in 1:l
        zero_wf!(ψ)
        zero_wf!(r)
        for j in 1:l
            axpby!(evecs[j,k], z[j], 1, ψ)         # ψ = Σ u_{jk} z_j
            axpby!(evecs[j,k], DdagDz[j], 1, r)     # r = Σ u_{jk} D†D z_j = D†D ψ_k
        end
        # r = D†D ψ_k - λ_k ψ_k
        axpby!(1, r, -evals[k], ψ)    # ψ = r - λ_k ψ → but we want r = D†Dψ - λψ
        # 위에서 axpby!(1,r,-λ,ψ) → ψ = 1*r + (-λ)*ψ = D†Dψ - λψ. 맞음
        conj_fermion!(VD[k], ψ)
    end

    # ∂/∂V 를 반환하려면 계수는 evecs[i,k] (conj 없음)
    for i in 1:l
        for k in 1:l
            axpby!(2*evecs[i,k]/m[k], VD[k], 1, delta[i])
        end
    end



end

function _gram_matrix_kernel!(
    G::CuDeviceMatrix{ComplexF64},
    X::CuDeviceVector{ComplexF64},
    Y::CuDeviceVector{ComplexF64},
    inner::Int32,
    n::Int32
)
    pair = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))

    i = (pair - Int32(1)) % n + Int32(1)
    j = (pair - Int32(1)) ÷ n + Int32(1)
    base_i = (i - Int32(1)) * inner
    base_j = (j - Int32(1)) * inner

    s = ComplexF64(0.0)
    p = tid
    while p <= inner
        @inbounds s += conj(X[base_i + p]) * Y[base_j + p]
        p += bsz
    end
    @inbounds sh[tid] = s
    sync_threads()

    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            @inbounds sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end

    if tid == Int32(1)
        @inbounds G[i, j] = sh[Int32(1)]
    end
    return nothing
end

function ritz_gram_matrices_gpu!(Agpu::CuArray{ComplexF64,2}, Bgpu::CuArray{ComplexF64,2},
                                 x::AbstractVector, Dx::AbstractVector)
    n = length(x)
    f1 = x[1].f
    NC, NG, bs, rs = size(f1, 1), size(f1, 2), size(f1, 3), size(f1, 4)
    Xbuf = get_shared_buf!(:ritz_x_all,  NC, NG, bs, rs, n)
    Dbuf = get_shared_buf!(:ritz_dx_all, NC, NG, bs, rs, n)
    pack_fermions!(Xbuf, x)
    pack_fermions!(Dbuf, Dx)
    inner = Int32(NC * NG * bs * rs)
    npairs = n * n
    @cuda threads=256 blocks=npairs shmem=(256 * sizeof(ComplexF64)) _gram_matrix_kernel!(
        Bgpu, reshape(Xbuf, :), reshape(Xbuf, :), inner, Int32(n))
    @cuda threads=256 blocks=npairs shmem=(256 * sizeof(ComplexF64)) _gram_matrix_kernel!(
        Agpu, reshape(Dbuf, :), reshape(Dbuf, :), inner, Int32(n))
    return Agpu, Bgpu
end

function ritz_gram_matrices_gpu!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64},
                                 x::AbstractVector, Dx::AbstractVector)
    n = length(x)
    Agpu = get_shared_matrix!(:ritz_A, n)
    Bgpu = get_shared_matrix!(:ritz_B, n)
    ritz_gram_matrices_gpu!(Agpu, Bgpu, x, Dx)
    copyto!(A, Agpu)
    copyto!(B, Bgpu)
    return A, B
end

# ============================================================
#  BlockScalar gradient: sum over spin/color (dims 1,2) — 0 allocation
#  PT layer: in-place conjugation kernel — 0 allocation
# ============================================================

function _sum_spin_color_kernel!(
    out::CuDeviceVector{ComplexF64},
    x::CuDeviceVector{ComplexF64},
    nc_ng::Int32,
    n_sites::Int32
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= n_sites
        s = ComplexF64(0.0)
        base = nc_ng * (tid - Int32(1))
        for j in Int32(1):nc_ng
            @inbounds s += x[base + j]
        end
        @inbounds out[tid] = s
    end
    nothing
end

function _sum_spin_color!(out::CuArray{ComplexF64}, x::CuArray{ComplexF64,4})
    nc_ng   = Int32(size(x, 1) * size(x, 2))
    n_sites = Int32(size(x, 3) * size(x, 4))
    @cuda threads=256 blocks=cld(Int(n_sites), 256) _sum_spin_color_kernel!(
        reshape(out, :), reshape(x, :), nc_ng, n_sites)
end
