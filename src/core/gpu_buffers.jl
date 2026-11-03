# Reusable GPU work buffers and cached device-pointer arrays, so that hot loops
# do not allocate.

# ---- Pack / Unpack fermion fields ----
# Packing n fields uses one kernel launch instead of n copyto! calls.
# The device pointers of the fields are cached under (buffer address, n), so
# only the first call for a set of fields allocates. They are stored as UInt64
# and reinterpreted as LLVMPtr inside the kernel. The whole pointer list is
# compared: fields that are freed and reallocated (recomputed activations) can
# get the old address for the first field and new ones for the others.
const _field_ptr_cache = Dict{Tuple{UInt64,Int}, Tuple{Vector{UInt64}, CuArray{UInt64,1}}}()

function _cached_field_ptrs!(buf::CuArray{ComplexF64,5}, fields)
    n    = length(fields)
    k    = (reinterpret(UInt64, pointer(buf)), n)
    ptrs = UInt64[reinterpret(UInt64, pointer(fields[i].f)) for i in 1:n]
    cached = get(_field_ptr_cache, k, nothing)
    if cached === nothing || cached[1] != ptrs
        _field_ptr_cache[k] = (ptrs, CuArray(ptrs))
    end
    return _field_ptr_cache[k][2]
end

# ============================================================
#  Shared GPU buffer pool
#  Work buffers such as result_all, d_buf, ci_all and hop_temp are only
#  needed until they are unpacked, so all layers share one of each.
# ============================================================
const _shared_gpu_pool = Dict{Tuple{Symbol,Int}, CuArray{ComplexF64,5}}()

"""
    get_shared_buf!(key, NC, NG, bs, rs, n) → CuArray{ComplexF64,5}

Shared 5D GPU work buffer of shape (NC, NG, bs, rs, n). The same (key, n)
always returns the same CuArray, so all PTLayers share it.
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

const _shared_matrix_pool = Dict{Tuple{Symbol,Int}, CuArray{ComplexF64,2}}()

function get_shared_matrix!(key::Symbol, n::Int)
    k = (key, n)
    buf = get(_shared_matrix_pool, k, nothing)
    if buf === nothing || size(buf) != (n, n)
        buf = CUDA.zeros(ComplexF64, n, n)
        _shared_matrix_pool[k] = buf
    end
    return buf
end
