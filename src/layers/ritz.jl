# Ritz_Layer: the Rayleigh-Ritz loss on the span of the input fields.

"""
    Ritz_Layer(D, activation=Identity(), Out=1)

Loss layer. For input fields `x[1:l]` it builds the Gram matrices

    A[i, j] = <D x[i], D x[j]>,    B[i, j] = <x[i], x[j]>

solves the generalized eigenproblem `A u_k = lambda_k B u_k` and returns
`sum_k lambda_k`, the sum of the Ritz values of `D^dagger D` on the span of the
inputs. Minimising it drives the span towards the low modes of `D`.

Backward pass, with `psi_k = sum_j u[j, k] z[j]` and `m_k = u_k^dagger B u_k`:

    delta[i] = 2 sum_k u[i, k] / m_k * conj((D^dagger D - lambda_k) psi_k)

The forward pass keeps `D^dagger D z[j]`, the eigenvalues and eigenvectors, so
the backward pass applies no Dirac operator.
"""
mutable struct Ritz_Layer <: Layer
    D::Any                    # Wilson Dirac operator (CPU or GPU)
    activation::ActivationFunction
    Out::Int
    # Kept from the forward pass for the backward pass
    _Dz::Vector{Any}          # D z[i]
    _DdagDz::Vector{Any}      # D^dagger D z[i]
    _VD::Vector{Any}          # backward work fields, one per input
    _work::Vector{Any}        # backward work fields [psi, r]
    _ritz_init::Bool
    # Gram matrices and the eigen decomposition (CPU arrays)
    _A::Matrix{ComplexF64}
    _B::Matrix{ComplexF64}
    _C::Matrix{ComplexF64}
    _Bfac::Matrix{ComplexF64}
    _evals::Vector{Float64}
    _evecs::Matrix{ComplexF64}
    _m::Vector{ComplexF64}    # m_k = u_k^dagger B u_k
    function Ritz_Layer(D::Any, activation::ActivationFunction= Identity(),Out =1)
        new(D, activation, Out,
            Any[], Any[], Any[], Any[], false,
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Float64[], Matrix{ComplexF64}(undef,0,0), ComplexF64[])
    end
end

# Printing the default way would touch GPU arrays outside a CUDA context.
function Base.show(io::IO, layer::Ritz_Layer)
    print(io, "Ritz_Layer(Out=$(layer.Out))")
end

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)

    # 1) Dx[i] = D x[i] and DdagDx[i] = D^dagger D x[i].
    need_reinit = !layer._ritz_init || length(layer._Dz) != l
    if !need_reinit
        # Reinitialize when the fields move between CPU and GPU.
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

    _is_gpu_ritz = x[1] isa WF3x1GPU
    if _is_gpu_ritz
        # Fused Wilson kernel, no allocation.
        _kappa = _dirac_kappa(layer.D)
        _U = layer.D.U
        @inbounds for i in 1:l
            fused_wilson_mul_gpu!(Dx[i], x[i], _U, _kappa, false)
            fused_wilson_mul_gpu!(DdagDx[i], Dx[i], _U, _kappa, true)
        end
    else
        @inbounds for i in 1:l
            mul!(Dx[i], layer.D, x[i])
            mul!(DdagDx[i], layer.D', Dx[i])
        end
    end

    # 2) Gram matrices A[i,j] = <Dx[i], Dx[j]> and B[i,j] = <x[i], x[j]>.
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
        # A and B stay on the GPU for the eigensolver.
        Agpu = get_shared_matrix!(:ritz_A, l)
        Bgpu = get_shared_matrix!(:ritz_B, l)
        ritz_gram_matrices_gpu!(Agpu, Bgpu, x, Dx)
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

    # 3) Report NaN or Inf entries before the eigensolver fails on them.
    if _is_gpu_ritz
        _report_ritz_nonfinite(Agpu, Bgpu, x, Dx, layer.D)
    else
        _report_ritz_nonfinite(A, B, x, Dx, layer.D)
    end

    # 4) Generalized eigen decomposition and m_k.
    if _is_gpu_ritz
        evals_out, evecs_out, _ = CUDA.CUSOLVER.hegvd!(1, 'V', 'U', Agpu, Bgpu)
        copyto!(layer._evals, evals_out)
        copyto!(layer._evecs, evecs_out)
        # hegvd! returns B-orthonormal eigenvectors, so m_k = 1.
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

# Prints where NaN or Inf values appear: in A or B, in an input field, in
# D times an input, or in the gauge links.
function _report_ritz_nonfinite(A, B, x, Dx, D)
    if any(isnan, A) || any(isinf, A) || any(isnan, B) || any(isinf, B)
        n_nanA = count(isnan, A); n_infA = count(isinf, A)
        n_nanB = count(isnan, B); n_infB = count(isinf, B)
        println("[Ritz NaN] A: $(n_nanA) NaN, $(n_infA) Inf / $(length(A)) | B: $(n_nanB) NaN, $(n_infB) Inf / $(length(B))")
        for i in eachindex(x)
            _xf = hasproperty(x[i], :f) ? x[i].f : nothing
            _dxf = hasproperty(Dx[i], :f) ? Dx[i].f : nothing
            if _xf !== nothing && (_xf isa AbstractArray) && (any(isnan, _xf) || any(isinf, _xf))
                println("[Ritz NaN] x[$i] contains NaN/Inf")
            end
            if _dxf !== nothing && (_dxf isa AbstractArray) && (any(isnan, _dxf) || any(isinf, _dxf))
                println("[Ritz NaN] Dx[$i] contains NaN/Inf")
            end
        end
        for (mu, ui) in enumerate(D.U)
            if hasproperty(ui, :U) && ui.U isa CuArray
                n_bad = count(x -> isnan(x) || isinf(x), ui.U)
                n_bad > 0 && println("[Ritz NaN] D.U[$mu]: $n_bad / $(length(ui.U)) NaN/Inf")
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

function DeltaRule!(layer::Ritz_Layer, ::Identity, z::Vector{WF3x1}, delta::Vector{WF3x1}, batch::Int)

    l = length(z)
    for i in 1:length(delta)
        zero_wf!(delta[i])
    end

    # Reuse the eigen decomposition and D^dagger D z from the forward pass.
    evals = layer._evals
    evecs = layer._evecs
    m     = layer._m
    DdagDz = layer._DdagDz

    need_vd_reinit = isempty(layer._VD) || length(layer._VD) != l
    if !need_vd_reinit
        need_vd_reinit = typeof(layer._VD[1]) != typeof(z[1])
    end
    if need_vd_reinit
        layer._VD = Vector{Any}(undef, l)
        for i in 1:l
            layer._VD[i] = _alloc_result_wf(z[1])
        end
        layer._work = Any[_alloc_result_wf(z[1]), _alloc_result_wf(z[1])]  # [psi, r]
    end

    VD = layer._VD
    ψ  = layer._work[1]
    r  = layer._work[2]

    # VD[k] = conj((D^dagger D - lambda_k) psi_k), using
    # D^dagger D psi_k = sum_j u[j,k] D^dagger D z[j].
    for k in 1:l
        zero_wf!(ψ)
        zero_wf!(r)
        for j in 1:l
            axpby!(evecs[j,k], z[j], 1, ψ)         # psi = sum_j u[j,k] z[j]
            axpby!(evecs[j,k], DdagDz[j], 1, r)     # r = D^dagger D psi
        end
        axpby!(1, r, -evals[k], ψ)                  # psi <- r - lambda_k psi
        conj_fermion!(VD[k], ψ)
    end

    for i in 1:l
        for k in 1:l
            axpby!(2*evecs[i,k]/m[k], VD[k], 1, delta[i])
        end
    end

end

"""
    DeltaRule(layer::Ritz_Layer, ::Identity, z, batch)

Allocating version of `DeltaRule!`: returns the input deltas.
"""
function DeltaRule(layer::Ritz_Layer, act::Identity, z::Vector{WF3x1}, batch::Int)
    result = Vector{WF3x1}(undef, length(z))
    for i in eachindex(result)
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end
    DeltaRule!(layer, act, z, result, batch)
    return result
end

# A loss layer has no weights.
function gradient(layer:: Ritz_Layer , ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion: moves D and drops the cached work fields
# ---------------------------------------------------------------------------

function _has_gpu(layer::Ritz_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
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

function _shared_layer_gauge(layer::Ritz_Layer)
    try
        return layer.D.U
    catch
        return nothing
    end
end

function _to_cpu(layer::Ritz_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
    return layer
end

function _to_gpu(layer::Ritz_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
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

# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------

# G[i, j] = sum_p conj(X_i[p]) Y_j[p]. One thread block per (i, j) pair,
# reduced in shared memory.
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

# A = Gram matrix of Dx and B = Gram matrix of x, computed on the GPU.
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

# Same, returning CPU matrices.
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
