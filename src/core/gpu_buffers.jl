# Reusable GPU work buffers and cached device-pointer arrays, so that hot loops
# do not allocate.

# ---- Pack / Unpack fermion fields ----
# n개의 serial copyto! 대신 single CUDA kernel launch 로 전체를 처리.
# 포인터 캐시: (buf_addr, n) 키로 CuArray{CuPtr} 를 재사용 — 첫 호출에만 할당.

# 포인터는 UInt64 로 저장 — kernel 내에서 LLVMPtr 로 reinterpret
const _field_ptr_cache = Dict{Tuple{UInt64,Int}, Tuple{UInt64, CuArray{UInt64,1}}}()

function _cached_field_ptrs!(buf::CuArray{ComplexF64,5}, fields)
    n  = length(fields)
    kb = reinterpret(UInt64, pointer(buf))
    p0 = reinterpret(UInt64, pointer(fields[1].f))
    k  = (kb, n)
    cached = get(_field_ptr_cache, k, nothing)
    if cached === nothing || cached[1] != p0
        _field_ptr_cache[k] = (p0, CuArray([reinterpret(UInt64, pointer(fields[i].f)) for i in 1:n]))
    end
    return _field_ptr_cache[k][2]
end

# ============================================================
#  Shared GPU buffer pool — layers 간 임시 버퍼 공유
#  (result_all, d_buf, ci_all, hop_temp 등 forward/backward 직후
#   unpack되어 재사용 가능한 버퍼를 공유하여 GPU 메모리 절약)
# ============================================================
const _shared_gpu_pool = Dict{Tuple{Symbol,Int}, CuArray{ComplexF64,5}}()

"""
    get_shared_buf!(key, NC, NG, bs, rs, n) → CuArray{ComplexF64,5}

Shared 5D GPU buffer. 동일 (key, n) 조합은 동일 CuArray를 반환.
PTLayer 인스턴스 간 임시 버퍼를 공유하여 GPU 메모리를 절약.
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
