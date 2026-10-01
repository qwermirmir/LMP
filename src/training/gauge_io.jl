# Reading gauge configurations for training: JLD2 files, the raw binary format
# and its converter, and the pinned staging buffers used to copy them to the GPU.

function _copy_debug_step(label::String, gc0, t0::UInt64; sync_gpu::Bool=false)
    sync_gpu && CUDA.synchronize()
    gc1 = Base.gc_num()
    t1 = time_ns()
    alloc_bytes = max(0, gc1.allocd - gc0.allocd)
    println("[copy_debug] ", label,
            " time=", round((t1 - t0) / 1e6; digits=3), "ms",
            " alloc=", round(alloc_bytes / 1024^2; digits=3), "MiB")
    return gc1, t1
end

function _pack_gauge_to_staging!(staging::Array, A, B)
    bi = A.blockinfo
    NC = A.NC
    @inbounds for r in 1:bi.rsize
        for b in 1:bi.blocksize
            ix = _cvt_br(b, r, bi.blocks[1], bi.blocks_s[1], bi.blocknumbers[1], bi.blocknumbers_s[1])
            iy = _cvt_br(b, r, bi.blocks[2], bi.blocks_s[2], bi.blocknumbers[2], bi.blocknumbers_s[2])
            iz = _cvt_br(b, r, bi.blocks[3], bi.blocks_s[3], bi.blocknumbers[3], bi.blocknumbers_s[3])
            it = _cvt_br(b, r, bi.blocks[4], bi.blocks_s[4], bi.blocknumbers[4], bi.blocknumbers_s[4])
            for ic in 1:NC, jc in 1:NC
                staging[jc, ic, b, r] = B[jc, ic, ix, iy, iz, it]
            end
        end
    end
    return staging
end

function _fast_substitute_U!(A, B, staging::Array; debug::Bool=false, label::String="")
    dbg_gc = debug ? Base.gc_num() : nothing
    dbg_t = debug ? time_ns() : UInt64(0)
    if debug
        dbg_gc, dbg_t = _copy_debug_step("$(label)setup", dbg_gc, dbg_t)
    end
    _pack_gauge_to_staging!(staging, A, B)
    if debug
        dbg_gc, dbg_t = _copy_debug_step("$(label)fill_staging", dbg_gc, dbg_t)
    end
    copyto!(A.U, staging)
    if debug
        dbg_gc, dbg_t = _copy_debug_step("$(label)copyto_gpu", dbg_gc, dbg_t; sync_gpu=true)
    end
end

load_gauge_cache(paths::Vector{String}) = [JLD2.load(path, "U") for path in paths]

const _RAW_GAUGE_MAGIC = UInt64(0x3155504347504d4c) # "LMPGCPU1" little-endian

function _make_gauge_staging(U_template)
    return [CUDA.pin(Array{ComplexF64}(undef, U_template[mu].NC, U_template[mu].NC,
                                       U_template[mu].blockinfo.blocksize,
                                       U_template[mu].blockinfo.rsize))
            for mu in 1:length(U_template)]
end

function _gauge_dims(U)
    return (Int(U.NC), Int(U.NX), Int(U.NY), Int(U.NZ), Int(U.NT))
end

function _write_raw_header(io, U_cpu)
    write(io, _RAW_GAUGE_MAGIC)
    write(io, Int64(length(U_cpu)))
    for mu in eachindex(U_cpu)
        nc, nx, ny, nz, nt = _gauge_dims(U_cpu[mu])
        write(io, Int64(nc))
        write(io, Int64(nx))
        write(io, Int64(ny))
        write(io, Int64(nz))
        write(io, Int64(nt))
    end
end

function _read_raw_header!(io, U_template)
    magic = read(io, UInt64)
    magic == _RAW_GAUGE_MAGIC || throw(ArgumentError("raw gauge magic mismatch"))
    dim = read(io, Int64)
    dim == length(U_template) || throw(ArgumentError("raw gauge Dim mismatch: file=$dim template=$(length(U_template))"))
    dims = Vector{NTuple{5,Int}}(undef, dim)
    for mu in eachindex(U_template)
        nc = read(io, Int64)
        nx = read(io, Int64)
        ny = read(io, Int64)
        nz = read(io, Int64)
        nt = read(io, Int64)
        nc == U_template[mu].NC || throw(ArgumentError("raw gauge NC mismatch at U[$mu]"))
        if hasproperty(U_template[mu], :NX)
            nx == U_template[mu].NX || throw(ArgumentError("raw gauge NX mismatch at U[$mu]"))
            ny == U_template[mu].NY || throw(ArgumentError("raw gauge NY mismatch at U[$mu]"))
            nz == U_template[mu].NZ || throw(ArgumentError("raw gauge NZ mismatch at U[$mu]"))
            nt == U_template[mu].NT || throw(ArgumentError("raw gauge NT mismatch at U[$mu]"))
        end
        dims[mu] = (Int(nc), Int(nx), Int(ny), Int(nz), Int(nt))
    end
    return dims
end

function _raw_path_for(path::String, raw_dir)
    base = splitext(basename(path))[1] * ".gbin"
    return raw_dir === nothing ? joinpath(dirname(path), base) : joinpath(raw_dir, base)
end

function _jld2_has_key(path::String, key::String)
    return jldopen(path, "r") do f
        haskey(f, key)
    end
end

function save_raw_gauge(jld2_path::String, raw_path::String, U_template)
    U_cpu = JLD2.load(jld2_path, "U")
    open(raw_path, "w") do io
        _write_raw_header(io, U_cpu)
        for mu in eachindex(U_cpu)
            write(io, U_cpu[mu].U)
        end
    end
    return raw_path
end

function convert_gauge_paths_to_raw(gauge_paths::Vector{String}, U_template;
                                    raw_dir=nothing, force::Bool=false,
                                    skip_missing_U::Bool=true)
    raw_dir !== nothing && mkpath(raw_dir)
    raw_paths = String[]
    for path in gauge_paths
        if !_jld2_has_key(path, "U")
            if skip_missing_U
                @warn "skip non-gauge JLD2 file without key \"U\"" path
                continue
            else
                throw(KeyError("U"))
            end
        end
        raw_path = _raw_path_for(path, raw_dir)
        if force || !isfile(raw_path)
            save_raw_gauge(path, raw_path, U_template)
        end
        push!(raw_paths, raw_path)
    end
    return raw_paths
end

function _make_raw_cpu_buffers(raw_path::String, U_template)
    open(raw_path, "r") do io
        dims = _read_raw_header!(io, U_template)
        return [CUDA.pin(Array{ComplexF64}(undef, nc, nc, nx, ny, nz, nt))
                for (nc, nx, ny, nz, nt) in dims]
    end
end

function _pack_raw_cpu_to_staging!(staging::Array, A, B::Array{ComplexF64,6})
    bi = A.blockinfo
    NC = A.NC
    @inbounds for r in 1:bi.rsize
        for b in 1:bi.blocksize
            ix = _cvt_br(b, r, bi.blocks[1], bi.blocks_s[1], bi.blocknumbers[1], bi.blocknumbers_s[1])
            iy = _cvt_br(b, r, bi.blocks[2], bi.blocks_s[2], bi.blocknumbers[2], bi.blocknumbers_s[2])
            iz = _cvt_br(b, r, bi.blocks[3], bi.blocks_s[3], bi.blocknumbers[3], bi.blocknumbers_s[3])
            it = _cvt_br(b, r, bi.blocks[4], bi.blocks_s[4], bi.blocknumbers[4], bi.blocknumbers_s[4])
            for ic in 1:NC, jc in 1:NC
                staging[jc, ic, b, r] = B[jc, ic, ix, iy, iz, it]
            end
        end
    end
    return staging
end

function _read_raw_gauge_to_gpu!(U_template, raw_path::String, staging, raw_cpu_buffers;
                                 debug::Bool=false, label::String="")
    gc0 = nothing
    t0 = UInt64(0)
    if debug
        gc0 = Base.gc_num()
        t0 = time_ns()
    end
    open(raw_path, "r") do io
        _read_raw_header!(io, U_template)
        for mu in eachindex(U_template)
            read!(io, raw_cpu_buffers[mu])
            _pack_raw_cpu_to_staging!(staging[mu], U_template[mu], raw_cpu_buffers[mu])
            copyto!(U_template[mu].U, staging[mu])
        end
    end
    debug && _copy_debug_step(label * "raw_read_copyto_gpu", gc0, t0; sync_gpu=true)
    return nothing
end
