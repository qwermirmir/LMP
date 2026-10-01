# The Adam optimizer: optimizer state, the global timestep, the update for
# spin-matrix weights and fermion fields on the CPU, and the GPU update kernels.

# ──── GPU-side Adam update ────

function _cudakernel_adam_complex!(
    W_gpu   ::CuDeviceArray{ComplexF64},
    m_gpu   ::CuDeviceArray{ComplexF64},
    v_gpu   ::CuDeviceArray{ComplexF64},
    grad_gpu::CuDeviceArray{ComplexF64},
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

Adam update of `W_gpu` on the GPU in one kernel launch, with no CPU transfer.
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

abstract type Opt end

mutable struct AdamState{T}
    m::T
    v::T
end

mutable struct Adam <: Opt
    α::Float64
    β1::Float64
    β2::Float64
    ε::Float64
    m::Dict{UInt, Spin_matrix}
    v::Dict{UInt, Spin_matrix}
    t::Int
    scalar_state::Dict{UInt, AdamState{Float64}}   # complex scalar weights, one entry per real and imaginary part
    wf_state::Dict{UInt, AdamState{WF3x1}}         # fermion-field parameters

    function Adam(α=0.001, β1=0.9, β2=0.999, ε=1e-8)
        new(α, β1, β2, ε,
            Dict{UInt, Spin_matrix}(),
            Dict{UInt, Spin_matrix}(),
            0,
            Dict{UInt, AdamState{Float64}}(),
            Dict{UInt, AdamState{WF3x1}}())
    end
end

"""Advance Adam's global timestep once for one optimizer sweep."""
function advance_step!(opt::Adam)
    opt.t += 1
    return opt.t
end

function complexProcess(param::T, opt:: Adam , grad ) where {T<:AbstractSpinMatrix}


    
    key = objectid(param)

    if !haskey(opt.m, key)
        opt.m[key] = Random_SpinM("zero")
        opt.v[key] = Random_SpinM("zero")
    end

    m = opt.m[key]
    v = opt.v[key]

    β1 = opt.β1
    β2 = opt.β2
    α  = opt.α
    ε  = opt.ε
    t  = opt.t
    bias1 = 1 / (1 - β1^t)
    bias2 = 1 / (1 - β2^t)

    sm_g = grad.sm
    sm_m = m.sm
    sm_v = v.sm
    sm_p = param.sm

    @inbounds for j in axes(sm_g, 2), i in axes(sm_g, 1)
        g_re = real(sm_g[i,j])
        g_im = imag(sm_g[i,j])

        mr = real(sm_m[i,j])
        mi = imag(sm_m[i,j])
        vr = real(sm_v[i,j])
        vi = imag(sm_v[i,j])

        # EMA update
        mr = β1 * mr + (1 - β1) * g_re
        vr = β2 * vr + (1 - β2) * g_re^2
        mi = β1 * mi + (1 - β1) * g_im
        vi = β2 * vi + (1 - β2) * g_im^2

        sm_m[i,j] = complex(mr, mi)
        sm_v[i,j] = complex(vr, vi)

        # bias-corrected update
        upd_re = α * (mr * bias1) / (sqrt(vr * bias2) + ε)
        upd_im = α * (mi * bias1) / (sqrt(vi * bias2) + ε)

        sm_p[i,j] -= complex(upd_re, -upd_im)
    end

end   

# One Adam step for a fermion-field parameter.
function wf_adam_step!(param::WF3x1, opt::Adam, grad::WF3x1)
    key = objectid(param)

    # Moments m and v, created as zero fields on first use.
    state = get!(opt.wf_state, key) do
        m0 = similar(param)
        v0 = similar(param)
        zero_wf!(m0)
        zero_wf!(v0)
        AdamState{WF3x1}(m0, v0)
    end
    m, v = state.m, state.v

    β1, β2, α, ε = opt.β1, opt.β2, opt.α, opt.ε
    t  = max(opt.t, 1)
    b1 = 1 - β1^t
    b2 = 1 - β2^t

    NC, NX, NY, NZ, NT, NG = param.NC, param.NX, param.NY, param.NZ, param.NT, param.NG

    @inbounds for ic = 1:NC, it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX, ig = 1:NG
        g  = grad[ic, ix, iy, iz, it, ig]
        mi = m[ic, ix, iy, iz, it, ig]
        vi = v[ic, ix, iy, iz, it, ig]

        g_re, g_im = real(g), imag(g)
        m_re, m_im = real(mi), imag(mi)
        v_re, v_im = real(vi), imag(vi)

        # First and second moments.
        m_re = β1*m_re + (1-β1)*g_re
        m_im = β1*m_im + (1-β1)*g_im
        v_re = β2*v_re + (1-β2)*(g_re*g_re)
        v_im = β2*v_im + (1-β2)*(g_im*g_im)

        m[ic, ix, iy, iz, it, ig] = complex(m_re, m_im)
        v[ic, ix, iy, iz, it, ig] = complex(v_re, v_im)

        # bias correction
        m_hat_re = m_re / b1
        m_hat_im = m_im / b1
        v_hat_re = v_re / b2
        v_hat_im = v_im / b2

        # The update.
        upd_re = α * m_hat_re / (sqrt(v_hat_re) + ε)
        upd_im = α * m_hat_im / (sqrt(v_hat_im) + ε)

        # param := param - (upd_re - i*upd_im)
        param[ic, ix, iy, iz, it, ig] -= (upd_re - im*upd_im)
    end

    return param
end

function kernel_wf_adam!(p, m, v, g, β1, β2, α, ε, b1, b2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(p)
        gi = g[i]
        mi = m[i]
        vi = v[i]

        g_re = real(gi); g_im = imag(gi)
        m_re = real(mi); m_im = imag(mi)
        v_re = real(vi); v_im = imag(vi)

        m_re = β1*m_re + (1-β1)*g_re
        m_im = β1*m_im + (1-β1)*g_im
        v_re = β2*v_re + (1-β2)*(g_re*g_re)
        v_im = β2*v_im + (1-β2)*(g_im*g_im)

        m[i] = ComplexF64(m_re, m_im)
        v[i] = ComplexF64(v_re, v_im)

        m_hat_re = m_re / b1
        m_hat_im = m_im / b1
        v_hat_re = v_re / b2
        v_hat_im = v_im / b2

        # Guard against rounding; v is never negative in exact arithmetic.
        v_hat_re = ifelse(v_hat_re < 0.0, 0.0, v_hat_re)
        v_hat_im = ifelse(v_hat_im < 0.0, 0.0, v_hat_im)

        upd_re = α * m_hat_re / (sqrt(v_hat_re) + ε)
        upd_im = α * m_hat_im / (sqrt(v_hat_im) + ε)

        # param -= (upd_re - i*upd_im) = Complex(upd_re, -upd_im)
        p[i] = p[i] - ComplexF64(upd_re, -upd_im)
    end
    return
end

function wf_adam_step!(param::WF3x1GPU, opt::Adam, grad::WF3x1GPU)
    key = objectid(param)

    state = get!(opt.wf_state, key) do
        # similar(param) may not be zeroed.
        m0 = similar(param)
        v0 = similar(param)
        zero_wf!(m0)
        zero_wf!(v0)
        AdamState{WF3x1}(m0, v0)
    end

    m = state.m::WF3x1GPU
    v = state.v::WF3x1GPU

    p = param.f
    g = grad.f
    mf = m.f
    vf = v.f

    t  = max(opt.t, 1)
    b1 = 1 - opt.β1^t
    b2 = 1 - opt.β2^t

    threads = 256
    blocks  = cld(length(p), threads)
    @cuda threads=threads blocks=blocks kernel_wf_adam!(p, mf, vf, g, opt.β1, opt.β2, opt.α, opt.ε, b1, b2)

    return param
end
