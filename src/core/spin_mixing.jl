# Batched GPU kernels that mix N input fields into M output fields with 4x4
# spin matrices, as DenseLinearLayer and PTLayer do. The weights are a
# 4 x 4 x (N*M) array with W[:, :, (i-1)*M + j] mapping input i to output j,
# and the fields are packed into (NC, NG, blocksize, rsize, n) arrays by
# pack_fermions!.
#
#   batched_forward!   y[j]  = sum_i W_ij x[i]
#   batched_delta!     c[i]  = sum_j W_ij^dagger conj(d[j])
#   batched_gradient!  G_ij  = sum over sites and colour of d[j] x[i]^T

# result[c,g,b,r,j] = sum_i sum_g2 W[g,g2,(i-1)*Out+j] * x[c,g2,b,r,i]
# One thread per site in a block (b), one block per block of sites (r).
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

Mixes the packed inputs `hopped_all` into all outputs with one kernel launch.
"""
function batched_forward!(result_all, W_gpu, hopped_all, blocksize, rsize, npaths, Out)
    NC = 3; NG = 4
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_forward!(
        result_all, W_gpu, hopped_all, NC, NG, npaths, Out)
end

# M[fg,xg,widx] = sum_{c,b,r} d[c,fg,b,r,j] * x[c,xg,b,r,i]
# with j = (widx-1)%Out + 1 and i = (widx-1)÷Out + 1. One thread block per
# matrix element, reduced in shared memory.
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

Computes every weight gradient into `grad_gpu` (`4 x 4 x NW`) with one kernel
launch, without copying to the CPU.
"""
function batched_gradient!(grad_gpu, d_all, hopped_all, blocksize, rsize, Out, npaths)
    NG = 4; NC = 3
    NW = npaths * Out
    threads = 256
    nblocks = NG * NG * NW
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_batched_gradient!(
        grad_gpu, d_all, hopped_all, NC, NG, blocksize, rsize, Out, npaths)
end

# ci[c,g,b,r,i] = sum_j sum_g2 Wadj[g,g2,(i-1)*Out+j] * conj(d[c,g2,b,r,j])
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

Computes `c_i = sum_j W_ij^dagger conj(d[j])` for every path with one kernel
launch. `d_all` holds `d` itself; the kernel takes the conjugate.
"""
function batched_delta!(ci_all, Wadj_gpu, d_all, blocksize, rsize, Out, npaths)
    NC = 3; NG = 4
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_delta!(
        ci_all, Wadj_gpu, d_all, NC, NG, Out, npaths)
end

function _cudakernel_adjoint_W!(Wadj, W, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n; return; end
    @inbounds for j in 1:4, i in 1:4
        Wadj[i, j, idx] = conj(W[j, i, idx])
    end
    return nothing
end

"""
    sync_Wadj_gpu!(Wadj_gpu, W_gpu)

Writes the conjugate transpose of every 4x4 weight into `Wadj_gpu`, on the GPU.
"""
function sync_Wadj_gpu!(Wadj_gpu, W_gpu)
    n = size(W_gpu, 3)
    threads = min(256, n)
    nblocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adjoint_W!(Wadj_gpu, W_gpu, n)
end
