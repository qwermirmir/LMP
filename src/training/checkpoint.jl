# Saving and loading training checkpoints (weights, Adam state and the Adam
# timestep).

"""
    save_checkpoint(path, layers, opt)

Saves W + Adam GPU state + opt.t without calling substitute_Layer!.
Layers remain on GPU throughout — no memory pool growth.
"""
function save_checkpoint(path::AbstractString, layers, opt::Adam)
    sync_layers!(layers)  # _W_gpu → _W_cpu_buf + layer.W (for BlockScalar)
    data = Dict{String,Any}("opt_t" => opt.t)

    for (i, layer) in enumerate(layers)
        if (layer isa PTLayer || layer isa DenseLinearLayer) && layer._W_gpu !== nothing
            # _W_cpu_buf is the (NG,NG,NW) staging buffer populated by sync_layers!
            data["layer_$(i)_W"] = layer._W_cpu_buf !== nothing ?
                                    copy(layer._W_cpu_buf) : Array(layer._W_gpu)
            if layer._adam_m_gpu !== nothing
                data["layer_$(i)_adam_m"] = Array(layer._adam_m_gpu)
                data["layer_$(i)_adam_v"] = Array(layer._adam_v_gpu)
            end
        elseif layer isa BlockScalar_Layer
            # layer.W already synced from GPU by sync_layers!
            data["layer_$(i)_W"] = copy(layer.W)
            cache = layer._gpu_w_buf
            if cache isa Dict && haskey(cache, :adam_m) && cache[:adam_m] !== nothing
                data["layer_$(i)_adam_m"] = Array(cache[:adam_m])
                data["layer_$(i)_adam_v"] = Array(cache[:adam_v])
            end
        elseif layer isa fermionfield_Layer
            for j in eachindex(layer.x)
                x = layer.x[j]
                data["layer_$(i)_fermion_x_$(j)"] = Array(x.f)
                if x isa WF3x1GPU
                    data["layer_$(i)_fermion_blocks_$(j)"] = Tuple(Int.(x.blockinfo.blocks))
                end
                state_key = objectid(x)
                if haskey(opt.wf_state, state_key)
                    state = opt.wf_state[state_key]
                    data["layer_$(i)_fermion_m_$(j)"] = Array(state.m.f)
                    data["layer_$(i)_fermion_v_$(j)"] = Array(state.v.f)
                end
            end
        end
    end

    JLD2.save(path, data)
    return path
end

_blocks_tuple(x) = (Int(x[1]), Int(x[2]), Int(x[3]), Int(x[4]))

function _checkpoint_source_blocks(data, key::String)
    meta_key = replace(key, "_x_" => "_blocks_")
    haskey(data, meta_key) || return nothing
    return _blocks_tuple(data[meta_key])
end

@inline _ckpt_cvt(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)

function _checkpoint_layout_copy_kernel!(
    dest, src, total::Int,
    sb1::Int, sb2::Int, sb3::Int, sb4::Int,
    db1::Int, db2::Int, db3::Int, db4::Int,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    sb_s1::Int, sb_s2::Int, sb_s3::Int, sb_s4::Int,
    sr_s1::Int, sr_s2::Int, sr_s3::Int, sr_s4::Int,
    db_s1::Int, db_s2::Int, db_s3::Int, db_s4::Int,
    dr_s1::Int, dr_s2::Int, dr_s3::Int, dr_s4::Int,
    drn1::Int, drn2::Int, drn3::Int, drn4::Int,
)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= total
        t = idx - 1
        c = (t % 3) + 1
        t = div(t, 3)
        a = (t % 4) + 1
        t = div(t, 4)
        bd = (t % (db1 * db2 * db3 * db4)) + 1
        rd = div(t, db1 * db2 * db3 * db4) + 1

        ix0 = _ckpt_cvt(bd, db1, db_s1) + _ckpt_cvt(rd, drn1, dr_s1) * db1
        iy0 = _ckpt_cvt(bd, db2, db_s2) + _ckpt_cvt(rd, drn2, dr_s2) * db2
        iz0 = _ckpt_cvt(bd, db3, db_s3) + _ckpt_cvt(rd, drn3, dr_s3) * db3
        it0 = _ckpt_cvt(bd, db4, db_s4) + _ckpt_cvt(rd, drn4, dr_s4) * db4

        bs = 1 +
             (ix0 % sb1) * sb_s1 +
             (iy0 % sb2) * sb_s2 +
             (iz0 % sb3) * sb_s3 +
             (it0 % sb4) * sb_s4
        rs = 1 +
             div(ix0, sb1) * sr_s1 +
             div(iy0, sb2) * sr_s2 +
             div(iz0, sb3) * sr_s3 +
             div(it0, sb4) * sr_s4

        @inbounds dest[c, a, bd, rd] = src[c, a, bs, rs]
    end
    return nothing
end

function _checkpoint_linear_canonical_copy_kernel!(
    dest, src, total::Int,
    db1::Int, db2::Int, db3::Int, db4::Int,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    db_s1::Int, db_s2::Int, db_s3::Int, db_s4::Int,
    dr_s1::Int, dr_s2::Int, dr_s3::Int, dr_s4::Int,
    drn1::Int, drn2::Int, drn3::Int, drn4::Int,
)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= total
        t = idx - 1
        c = (t % 3) + 1
        t = div(t, 3)
        a = (t % 4) + 1
        t = div(t, 4)
        bd = (t % (db1 * db2 * db3 * db4)) + 1
        rd = div(t, db1 * db2 * db3 * db4) + 1

        ix0 = _ckpt_cvt(bd, db1, db_s1) + _ckpt_cvt(rd, drn1, dr_s1) * db1
        iy0 = _ckpt_cvt(bd, db2, db_s2) + _ckpt_cvt(rd, drn2, dr_s2) * db2
        iz0 = _ckpt_cvt(bd, db3, db_s3) + _ckpt_cvt(rd, drn3, dr_s3) * db3
        it0 = _ckpt_cvt(bd, db4, db_s4) + _ckpt_cvt(rd, drn4, dr_s4) * db4
        site = 1 + ix0 + iy0 * NX + iz0 * NX * NY + it0 * NX * NY * NZ
        src_idx = c + 3 * (a - 1) + 12 * (site - 1)

        @inbounds dest[c, a, bd, rd] = src[src_idx]
    end
    return nothing
end

function _copy_checkpoint_fermion_linear_canonical!(x::WF3x1GPU, arr)
    length(arr) == length(x.f) || error("Checkpoint fermion length mismatch: checkpoint has $(length(arr)), field has $(length(x.f)).")
    src = CuArray{ComplexF64}(reshape(arr, :))
    dst_blocks = _blocks_tuple(x.blockinfo.blocks)
    db1, db2, db3, db4 = dst_blocks
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    drn1, drn2, drn3, drn4 = NX ÷ db1, NY ÷ db2, NZ ÷ db3, NT ÷ db4
    total = length(x.f)
    threads = 256
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _checkpoint_linear_canonical_copy_kernel!(
        x.f, src, total,
        db1, db2, db3, db4,
        NX, NY, NZ, NT,
        1, db1, db1 * db2, db1 * db2 * db3,
        1, drn1, drn1 * drn2, drn1 * drn2 * drn3,
        drn1, drn2, drn3, drn4,
    )
    return x
end

function _copy_checkpoint_fermion_array!(x::WF3x1GPU, arr, src_blocks::NTuple{4,Int})
    dst_blocks = _blocks_tuple(x.blockinfo.blocks)
    if src_blocks == dst_blocks && size(arr) == size(x.f)
        copyto!(x.f, arr)
        return x
    end

    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    src_blocksize = prod(src_blocks)
    src_rsize = (NX * NY * NZ * NT) ÷ src_blocksize
    if size(arr, 3) != src_blocksize || size(arr, 4) != src_rsize
        error("Checkpoint fermion layout mismatch: source blocks=$src_blocks imply f size " *
              "(3,4,$src_blocksize,$src_rsize), but checkpoint has $(size(arr)).")
    end

    sb1, sb2, sb3, sb4 = src_blocks
    db1, db2, db3, db4 = dst_blocks
    srn1, srn2, srn3, srn4 = NX ÷ sb1, NY ÷ sb2, NZ ÷ sb3, NT ÷ sb4
    drn1, drn2, drn3, drn4 = NX ÷ db1, NY ÷ db2, NZ ÷ db3, NT ÷ db4
    src = CuArray{ComplexF64,4}(arr)
    total = length(x.f)
    threads = 256
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _checkpoint_layout_copy_kernel!(
        x.f, src, total,
        sb1, sb2, sb3, sb4,
        db1, db2, db3, db4,
        NX, NY, NZ, NT,
        1, sb1, sb1 * sb2, sb1 * sb2 * sb3,
        1, srn1, srn1 * srn2, srn1 * srn2 * srn3,
        1, db1, db1 * db2, db1 * db2 * db3,
        1, drn1, drn1 * drn2, drn1 * drn2 * drn3,
        drn1, drn2, drn3, drn4,
    )
    return x
end

function _copy_checkpoint_fermion_array!(x::WF3x1CPU, arr, src_blocks)
    size(arr) == size(x.f) || error("Cannot layout-convert fermion checkpoint into CPU field without CPU canonical conversion.")
    copyto!(x.f, arr)
    return x
end

function _load_checkpoint_fermion_field!(x::WF3x1GPU, data, key::String)
    arr = data[key]
    src_blocks = _checkpoint_source_blocks(data, key)
    if src_blocks === nothing
        return _copy_checkpoint_fermion_linear_canonical!(x, arr)
    end
    return _copy_checkpoint_fermion_array!(x, arr, src_blocks)
end

function _load_checkpoint_fermion_field!(x::WF3x1CPU, data, key::String)
    return _copy_checkpoint_fermion_array!(x, data[key], nothing)
end

function _load_checkpoint_fermion_adam_field!(x::WF3x1GPU, arr, data, key_x::String)
    src_blocks = _checkpoint_source_blocks(data, key_x)
    if src_blocks === nothing
        return _copy_checkpoint_fermion_linear_canonical!(x, arr)
    end
    return _copy_checkpoint_fermion_array!(x, arr, src_blocks)
end

function _load_checkpoint_fermion_adam_field!(x::WF3x1CPU, arr, data, key_x::String)
    return _copy_checkpoint_fermion_array!(x, arr, nothing)
end

function _lattice_dims_from_layers(layers)
    for layer in layers
        if layer isa fermionfield_Layer && !isempty(layer.x)
            x = layer.x[1]
            return (Int(x.NX), Int(x.NY), Int(x.NZ), Int(x.NT))
        end
    end
    return nothing
end

function _infer_block_r(nblocks::Int, dims::NTuple{4,Int})
    volume = prod(dims)
    volume % nblocks == 0 || return nothing
    block_volume = volume ÷ nblocks
    r = round(Int, block_volume^(1 / 4))
    return r^4 == block_volume ? r : nothing
end

@inline function _block_id(bx::Int, by::Int, bz::Int, bt::Int, NBX::Int, NBY::Int, NBZ::Int)
    return bx + NBX * ((by - 1) + NBY * ((bz - 1) + NBZ * (bt - 1)))
end

function _convert_blockscalar_weights(arr::AbstractMatrix{ComplexF64},
                                      dims::NTuple{4,Int},
                                      src_r::Int,
                                      dst_r::Int)
    src_r == dst_r && return copy(arr)
    NX, NY, NZ, NT = dims
    src_NBX, src_NBY, src_NBZ, src_NBT = NX ÷ src_r, NY ÷ src_r, NZ ÷ src_r, NT ÷ src_r
    dst_NBX, dst_NBY, dst_NBZ, dst_NBT = NX ÷ dst_r, NY ÷ dst_r, NZ ÷ dst_r, NT ÷ dst_r
    src_nblocks = src_NBX * src_NBY * src_NBZ * src_NBT
    dst_nblocks = dst_NBX * dst_NBY * dst_NBZ * dst_NBT
    size(arr, 1) == src_nblocks || error("source block count does not match inferred source r")

    out = zeros(ComplexF64, dst_nblocks, size(arr, 2))
    counts = zeros(Int, dst_nblocks)

    if src_r < dst_r
        dst_r % src_r == 0 || error("source r=$src_r does not divide destination r=$dst_r")
        ratio = dst_r ÷ src_r
        @inbounds for sbt in 1:src_NBT, sbz in 1:src_NBZ, sby in 1:src_NBY, sbx in 1:src_NBX
            dbx = (sbx - 1) ÷ ratio + 1
            dby = (sby - 1) ÷ ratio + 1
            dbz = (sbz - 1) ÷ ratio + 1
            dbt = (sbt - 1) ÷ ratio + 1
            src_bid = _block_id(sbx, sby, sbz, sbt, src_NBX, src_NBY, src_NBZ)
            dst_bid = _block_id(dbx, dby, dbz, dbt, dst_NBX, dst_NBY, dst_NBZ)
            out[dst_bid, :] .+= arr[src_bid, :]
            counts[dst_bid] += 1
        end
        @inbounds for bid in eachindex(counts)
            out[bid, :] ./= counts[bid]
        end
    else
        src_r % dst_r == 0 || error("destination r=$dst_r does not divide source r=$src_r")
        ratio = src_r ÷ dst_r
        @inbounds for dbt in 1:dst_NBT, dbz in 1:dst_NBZ, dby in 1:dst_NBY, dbx in 1:dst_NBX
            sbx = (dbx - 1) ÷ ratio + 1
            sby = (dby - 1) ÷ ratio + 1
            sbz = (dbz - 1) ÷ ratio + 1
            sbt = (dbt - 1) ÷ ratio + 1
            src_bid = _block_id(sbx, sby, sbz, sbt, src_NBX, src_NBY, src_NBZ)
            dst_bid = _block_id(dbx, dby, dbz, dbt, dst_NBX, dst_NBY, dst_NBZ)
            out[dst_bid, :] .= arr[src_bid, :]
        end
    end

    return out
end

function _maybe_convert_blockscalar_checkpoint(arr::AbstractMatrix{ComplexF64},
                                               layer::BlockScalar_Layer,
                                               dims)
    size(arr) == size(layer.W) && return copy(arr)
    dims === nothing && return nothing
    src_r = _infer_block_r(size(arr, 1), dims)
    src_r === nothing && return nothing
    try
        return _convert_blockscalar_weights(arr, dims, src_r, layer.r)
    catch
        return nothing
    end
end

"""
    load_checkpoint!(path, layers, opt)

Restores W + Adam state into live GPU layers from a W-only checkpoint.
No substitute_Layer! needed — layers stay on GPU.
Falls back silently if a layer's key is absent (partial checkpoint).
"""
function load_checkpoint!(path::AbstractString, layers, opt::Adam)
    data = JLD2.load(path)
    opt.t = Int(get(data, "opt_t", 0))
    lattice_dims = _lattice_dims_from_layers(layers)

    for (i, layer) in enumerate(layers)
        if layer isa PTLayer || layer isa DenseLinearLayer
            key_W = "layer_$(i)_W"
            if haskey(data, key_W)
                w_arr = data[key_W]  # Array{ComplexF64,3} (NG,NG,NW)
                expected = layer._W_gpu === nothing ? size(w_arr) : size(layer._W_gpu)
                if expected != size(w_arr)
                    @warn "Skipping checkpoint weight with incompatible shape" layer=i key=key_W expected got=size(w_arr)
                    continue
                end
                if layer._W_gpu === nothing
                    layer._W_gpu = CuArray{ComplexF64,3}(w_arr)
                else
                    copyto!(layer._W_gpu, w_arr)
                end
                if layer._W_cpu_buf === nothing
                    layer._W_cpu_buf = copy(w_arr)
                else
                    copyto!(layer._W_cpu_buf, w_arr)
                end
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
            key_m = "layer_$(i)_adam_m"
            if haskey(data, key_m) && size(data[key_m]) == size(layer._W_gpu)
                layer._adam_m_gpu = CuArray{ComplexF64,3}(data[key_m])
                layer._adam_v_gpu = CuArray{ComplexF64,3}(data["layer_$(i)_adam_v"])
            end
        elseif layer isa BlockScalar_Layer
            key_W = "layer_$(i)_W"
            if haskey(data, key_W)
                w_arr = _maybe_convert_blockscalar_checkpoint(data[key_W], layer, lattice_dims)
                if w_arr === nothing
                    @warn "Skipping checkpoint BlockScalar weight with incompatible shape" layer=i key=key_W expected=size(layer.W) got=size(data[key_W])
                    continue
                end
                copyto!(layer.W, w_arr)
                cache = layer._gpu_w_buf
                if cache isa Dict && haskey(cache, :W) && cache[:W] !== nothing
                    copyto!(cache[:W], layer.W)
                end
            end
            key_m = "layer_$(i)_adam_m"
            if haskey(data, key_m)
                m_arr = _maybe_convert_blockscalar_checkpoint(data[key_m], layer, lattice_dims)
                v_arr = _maybe_convert_blockscalar_checkpoint(data["layer_$(i)_adam_v"], layer, lattice_dims)
                cache = layer._gpu_w_buf
                if cache isa Dict && haskey(cache, :W) && cache[:W] !== nothing
                    m_arr !== nothing && (cache[:adam_m] = CuArray{ComplexF64}(m_arr))
                    v_arr !== nothing && (cache[:adam_v] = CuArray{ComplexF64}(v_arr))
                elseif m_arr !== nothing && v_arr !== nothing
                    layer._gpu_w_buf = Dict{Symbol,Any}(
                        :adam_m => m_arr,
                        :adam_v => v_arr,
                    )
                end
            end
        elseif layer isa fermionfield_Layer
            for j in eachindex(layer.x)
                key_x = "layer_$(i)_fermion_x_$(j)"
                haskey(data, key_x) || continue
                x = layer.x[j]
                _load_checkpoint_fermion_field!(x, data, key_x)

                key_m = "layer_$(i)_fermion_m_$(j)"
                if haskey(data, key_m)
                    m = similar(x)
                    v = similar(x)
                    _load_checkpoint_fermion_adam_field!(m, data[key_m], data, key_x)
                    _load_checkpoint_fermion_adam_field!(v, data["layer_$(i)_fermion_v_$(j)"], data, key_x)
                    opt.wf_state[objectid(x)] = AdamState{WF3x1}(m, v)
                end
            end
        end
    end

    CUDA.synchronize()
    return layers
end

