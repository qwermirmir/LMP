# Operations on Wilson fermion fields: arithmetic, inner products and norms,
# spin-matrix application, site-wise spin-colour contractions, copying,
# conjugation and packing into batched GPU arrays.

ndims(::Type{WF3x1}) = 6

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

    # Copy the entries of M into locals.
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

    # Copy the entries of M into locals.
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

function broadcast!(::typeof(/), x::WF3x1, y::Number)
    for ic in 1:x.NC, it in 1:x.NT, iz in 1:x.NZ, iy in 1:x.NY, ix in 1:x.NX
        for i in 1:4
            x[ic, ix, iy, iz, it, i] /= y
        end
    end
    return x
end

function Base.conj(x::WF3x1CPU)
    f=deepcopy(x)
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

function Base.:*(W::Vector{Spin_matrix_GPU}, x::Vector{WF3x1GPU})
    if length(W) != length(x)
        throw(ArgumentError("W and x must have the same length."))
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
        throw(ArgumentError("W and x must have the same length."))
    end

    result = similar(x)

    for i in 1:length(W)
        if x[i] isa WF3x1GPU
            result[i] = W[i] * x[i]
        else
            throw(ArgumentError("unsupported field type: $(typeof(x[i]))"))
        end
    end

    return result
end

function norm2(x::WF3x1GPU)
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

# Kernels for applying a spin matrix to a GPU field.
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

function conj_fermion!(out::WF3x1GPU, inp::WF3x1GPU)
    total = length(inp.f)
    threads = 256
    blocks  = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks kernel_conj_1D!(out.f, inp.f)
    return out
end

function conj_fermion!(x::WF3x1CPU,y::WF3x1CPU)
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

function Base.:*(M::Spin_matrix_GPU, x::WF3x1GPU)
    f  = deepcopy(x)
    Ms = SMatrix{4,4,ComplexF64}(M.sm)   # passed to the kernel by value

    CUDA.@cuda threads=x.blockinfo.blocksize blocks=x.blockinfo.rsize cudakernel_spinmat_mul!(f.f, Ms, x.f)

    return f
end

function Base.:+(a::WF3x1GPU,b::WF3x1GPU)

    f =deepcopy(b)



    axpby!(1,a,1,f)

    return f
end 

function Base.:-(a::WF3x1GPU,b::WF3x1GPU)

    y=deepcopy(b)



    axpby!(1,a,-1,y)

    return y
end 

function Base.conj(W::Spin_matrix_GPU)
    return Spin_matrix_GPU(conj(W.sm))
end

# Batched Matmul: N pairs (Q, K) at once, with one GPU-to-CPU copy.
function kernel_matmul_batched!(
    M_all::CuDeviceArray{ComplexF64,3},   # (NG, NG, batch)
    Q_ptrs::CuDeviceArray{ComplexF64,4},
    K_ptrs::CuDeviceArray{ComplexF64,4},
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

`Matmul!` for N pairs (Q, K) with a single GPU-to-CPU copy.
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

    # One GPU-to-CPU copy.
    M_cpu = Array(M_all)

    result = Vector{Spin_matrix_GPU}(undef, n)
    @inbounds for idx in 1:n
        result[idx] = Spin_matrix_GPU(MMatrix{4,4,ComplexF64}(@view M_cpu[:,:,idx]))
    end
    return result
end

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

    # Partial sum of this thread.
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

    # Reduction within the block in shared memory.
    sh = CUDA.CuStaticSharedArray(ComplexF64, 256)  # assumes 256 threads
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

    # Only a 4x4 matrix is copied, but the copy synchronises.
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

# GPU: dest(::WF3x1GPU) <- src.parent.fshifted  (device-to-device copy)
function substitute_fermion!(
    dest::WF3x1GPU,
    src::Shifted_fermionfields_4D_accelerator{NC,<:WF3x1GPU}
) where {NC}
    parent = src.parent
    @assert parent.fshifted !== nothing "the field was created with noshiftfield=true and has no fshifted buffer."
    copyto!(dest.f, parent.fshifted)
    return dest
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

    z0 = zero(typeof(x[1,1,1,1,1,1]))

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

function _pack_kernel!(dst, ptrs_u64, inner::Int32)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    fid = Int32(blockIdx().y)
    if tid <= inner
        ptr = reinterpret(Core.LLVMPtr{ComplexF64, 1}, ptrs_u64[fid])
        @inbounds dst[(fid - Int32(1)) * inner + tid] = unsafe_load(ptr, tid)
    end
    return nothing
end

"""Vector{WF3x1GPU} → CuArray{ComplexF64, 5} (NC,NG,bs,rs,n)"""
function pack_fermions!(buf::CuArray{ComplexF64, 5}, fields::AbstractVector)
    n = length(fields); n == 0 && return
    inner = size(buf,1) * size(buf,2) * size(buf,3) * size(buf,4)
    if n >= 32
        ptrs   = _cached_field_ptrs!(buf, fields)
        nblock = cld(inner, 256)
        @cuda threads=256 blocks=(nblock, Int(n)) _pack_kernel!(reshape(buf, :), ptrs, Int32(inner))
    else
        @inbounds for i in 1:n
            copyto!(buf, (i-1)*inner + 1, fields[i].f, 1, inner)
        end
    end
end

"""Copies slice i of a packed (NC, NG, bs, rs, n) array into fields[i].f."""
function unpack_fermions!(fields::AbstractVector, buf::CuArray{ComplexF64, 5})
    len = size(buf,1) * size(buf,2) * size(buf,3) * size(buf,4)
    @inbounds for j in eachindex(fields)
        copyto!(fields[j].f, 1, buf, (j-1)*len + 1, len)
    end
end

# ============================================================
#  deepcopy of a GPU field points its fshifted buffer at one scratch array
#  shared by every copy of the same size. fshifted only holds the shifted copy
#  of a field between a LatticeDiracOperators shift and the multiplication that
#  reads it, so copies can share it as long as no single operation needs the
#  shifted data of two copies at once; the LMP GPU kernels never use it. On a
#  16^3x32 lattice this saves 24 MB per buffer, about 18.6 GB for 31 layers x
#  25 fields. A smaller placeholder would make LatticeDiracOperators' mul!
#  write past it and return wrong results.
# ============================================================
const _SHARED_FSHIFTED = Dict{Any,CuArray{ComplexF64,4,CUDA.DeviceMemory}}()

function _shared_fshifted(x::WF3x1GPU)
    dims = size(x.f)
    return get!(() -> CUDA.zeros(ComplexF64, dims), _SHARED_FSHIFTED, (CUDA.device(), dims))
end

_is_shared_fshifted(buffer) = any(shared -> shared === buffer, values(_SHARED_FSHIFTED))

function Base.deepcopy_internal(x::WF3x1GPU, dict::IdDict)
    haskey(dict, x) && return dict[x]::WF3x1GPU
    # immutable struct: collect all fields first, then construct via jl_new_structv
    nf = nfields(x)
    fields = Vector{Any}(undef, nf)
    for i in 1:nf
        if fieldname(WF3x1GPU, i) === :fshifted
            fields[i] = _shared_fshifted(x)
        elseif isdefined(x, i)
            fields[i] = Base.deepcopy_internal(getfield(x, i), dict)
        end
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), WF3x1GPU, fields, UInt32(nf))::WF3x1GPU
    dict[x] = y
    return y
end

function _alloc_result_wf(x::WF3x1GPU)
    return deepcopy(x)  # shares fshifted, see the deepcopy_internal method above
end

_alloc_result_wf(x::WF3x1CPU) = deepcopy(x)

function _inplace_conj_kernel!(x::CuDeviceVector{ComplexF64}, n::Int32)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= n
        @inbounds x[tid] = conj(x[tid])
    end
    nothing
end

function inplace_conj!(x::CuArray{ComplexF64})
    n = Int32(length(x))
    @cuda threads=256 blocks=cld(Int(n), 256) _inplace_conj_kernel!(reshape(x, :), n)
end
