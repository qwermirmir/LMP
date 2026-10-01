# Moving a model between CPU and GPU (substitute_Layer! and the per-type
# _to_gpu/_to_cpu conversions), and pointing a model at a new gauge
# configuration (refresh_model!).

"""
    sync_layers!(layers)

저장/substitute_Layer! 전에 호출. _W_gpu → W (CPU) 동기화.
"""
function sync_layers_to_gpu!(layers)
    for layer in layers
        if layer isa PTLayer && !isempty(layer.W) && layer.W[1] isa Spin_matrix_GPU
            NW = length(layer.W)
            NG = 4
            if layer._W_gpu === nothing || size(layer._W_gpu, 3) != NW
                layer._W_gpu     = CUDA.zeros(ComplexF64, NG, NG, NW)
                layer._W_cpu_buf = Array{ComplexF64}(undef, NG, NG, NW)
            end
            if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
                layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            end
            sync_W_to_gpu!(layer._W_gpu, layer.W)
        end
    end
    return layers
end

function sync_layers!(layers)
    for layer in layers
        if layer isa PTLayer && layer._W_gpu !== nothing
            if layer._W_cpu_buf !== nothing
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            else
                layer._W_cpu_buf = Array{ComplexF64}(undef, size(layer._W_gpu)...)
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
        elseif layer isa DenseLinearLayer && layer._W_gpu !== nothing
            if layer._W_cpu_buf !== nothing
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            else
                layer._W_cpu_buf = Array{ComplexF64}(undef, size(layer._W_gpu)...)
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
        elseif layer isa BlockScalar_Layer && layer._gpu_w_buf isa Dict && haskey(layer._gpu_w_buf, :W)
            copyto!(layer.W, layer._gpu_w_buf[:W])
        end
    end
end

function substitute_Layer!(layers::Vector{<:Layer})
    # Toggle all layers between GPU and CPU representations.
    if _is_gpu_layer(layers)
        # Preserve GPU block layout so a later CPU -> GPU conversion can reuse it.
        _gpu_blocks = _extract_gpu_blocks(layers)
        for i in eachindex(layers)
            layers[i] = _to_cpu(layers[i])
        end
        # Save blocks after converting to CPU.
        if _gpu_blocks !== nothing
            _saved_blocks[] = _gpu_blocks
        end
        _force_cuda_reclaim!()
    else
        # Convert CPU layers back to GPU, reusing the saved block layout when available.
        _force_cuda_reclaim!()
        blks = _saved_blocks[]
        shared_U_gpu = _shared_gpu_gauge_for_layers(layers, blks)
        for i in eachindex(layers)
            layers[i] = _to_gpu(layers[i], blks, shared_U_gpu)
        end
        CUDA.synchronize()
    end
    return layers
end

function substitute_Layer!(layers::Vector{<:Layer}, blocks)
    _saved_blocks[] = blocks
    if _is_gpu_layer(layers)
        for i in eachindex(layers)
            layers[i] = _to_cpu(layers[i])
        end
        _force_cuda_reclaim!()
    else
        _force_cuda_reclaim!()
    end
    shared_U_gpu = _shared_gpu_gauge_for_layers(layers, blocks)
    for i in eachindex(layers)
        layers[i] = _to_gpu(layers[i], blocks, shared_U_gpu)
    end
    CUDA.synchronize()
    return layers
end

# Saved GPU block layout used when toggling layers back to GPU.
const _saved_blocks = Ref{Any}(nothing)

# Extract GPU gauge blocks from the first layer that owns them.
function _extract_gpu_blocks(layers::Vector{<:Layer})
    for layer in layers
        blks = _get_blocks(layer)
        if blks !== nothing
            return blks
        end
    end
    return nothing
end

_get_blocks(::Layer) = nothing

# --- GPU layer detection ---
function _is_gpu_layer(layers::Vector{<:Layer})
    for layer in layers
        if _has_gpu(layer)
            return true
        end
    end
    return false
end

_has_gpu(::Layer) = false

# --- GPU -> CPU gauge field ---
function _to_cpu_gauge(U::Vector{G}) where {G<:AbstractGaugefields}

        NC = U[1].NC
        NX = U[1].NX; NY = U[1].NY; NZ = U[1].NZ; NT = U[1].NT
        Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
        try
            substitute_U!(Ucpu, U)
        catch
            @warn "Failed to copy GPU gauge to CPU; using identity gauge. Check layer.U."
        end
        return Ucpu

end

# --- GPU -> CPU Spin matrix ---
_to_cpu_W(W::Vector{Spin_matrix_GPU}) = Convertsm(W)

_to_cpu_W(W::Vector{Spin_matrix}) = W

_to_cpu_W(W::Vector{T}) where {T<:AbstractArray{ComplexF64,2}} = W

# --- GPU -> CPU fermion field ---
# Convert GPU fermions, including JLD2-loaded fields, to CPU fermions.
function _to_cpu_fermion(x::WF3x1GPU)
    NC = x.NC
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    xcpu = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
    try
        substitute_fermion!(xcpu, x)
    catch
        @warn "Failed to copy GPU fermion to CPU; returning zero CPU fermion."
    end
    return xcpu
end

_to_cpu_fermion(x::WF3x1CPU) = x

function _to_cpu_fermions(xs::Vector{<:WF3x1})
    if isempty(xs) || xs[1] isa WF3x1CPU
        return xs
    end
    NC = xs[1].NC
    NX = xs[1].NX; NY = xs[1].NY; NZ = xs[1].NZ; NT = xs[1].NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    result = Vector{WF3x1CPU}(undef, length(xs))
    warned = false
    for i in eachindex(xs)
        result[i] = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
        if xs[i] isa WF3x1GPU
            try
                substitute_fermion!(result[i], xs[i])
            catch
                if !warned
                    @warn "Failed to copy GPU fermion to CPU; returning zero CPU fermion."
                    warned = true
                end
            end
        elseif xs[i] isa WF3x1CPU
            substitute_fermion!(result[i], xs[i])
        end
    end
    return result
end

# --- Generic GPU -> CPU layer fallback ---
_to_cpu(layer::Layer) = layer

# --- Dirac operator GPU -> CPU (D.U gauge field) ---
function _to_cpu_dirac(D)
    U = D.U
    if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        Ucpu = _to_cpu_gauge(U)
        xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson", nowing=true)
        params = Dict{String,Any}()
        params["Dirac_operator"] = "Wilson"
        params["κ"] = _dirac_kappa(D)
        params["eps_CG"] = D.eps_CG
        params["faster version"] = isa(D, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator_faster)
        return Dirac_operator(Ucpu, xcpu, params)
    else
        return D
    end
end

# ============================================================
# CPU -> GPU
# ============================================================

# --- CPU -> GPU gauge field ---
function _to_gpu_gauge(U::Vector{G}; blocks=nothing) where {G<:AbstractGaugefields}
    if G <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return U # already on GPU
    end
    NC = U[1].NC
    NX = U[1].NX; NY = U[1].NY; NZ = U[1].NZ; NT = U[1].NT
    if blocks !== nothing
        Ugpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Ugpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    substitute_U!(Ugpu, U)
    return Ugpu
end

# --- CPU -> GPU Spin matrix ---
_to_gpu_W(W::Vector{Spin_matrix_GPU}) = W

_to_gpu_W(W::Vector{Spin_matrix}) = [Spin_matrix_GPU(w) for w in W]

function _to_gpu_fermions(xs::Vector{<:WF3x1}; blocks=nothing)
    if isempty(xs) || xs[1] isa WF3x1GPU
        return xs
    end
    NC = xs[1].NC
    NX = xs[1].NX; NY = xs[1].NY; NZ = xs[1].NZ; NT = xs[1].NT
    if blocks !== nothing
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    result = Vector{WF3x1GPU}(undef, length(xs))
    for i in eachindex(xs)
        result[i] = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
        substitute_fermion!(result[i], xs[i])
    end
    return result
end

# --- Dirac operator CPU -> GPU ---
function _to_gpu_dirac(D; blocks=nothing, shared_U=nothing)
    U = D.U
    if shared_U === nothing && eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return D  # already on GPU
    end
    Ugpu = shared_U === nothing ? _to_gpu_gauge(U; blocks=blocks) : shared_U
    xgpu = Initialize_pseudofermion_fields(Ugpu[1], "Wilson", nowing=true)
    params = Dict{String,Any}()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = _dirac_kappa(D)
    params["eps_CG"] = D.eps_CG
    params["faster version"] = isa(D, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator_faster)
    return Dirac_operator(Ugpu, xgpu, params)
end

# --- Generic CPU -> GPU layer fallback ---
_to_gpu(layer::Layer, blocks=nothing) = layer

_to_gpu(layer::Layer, blocks, shared_U) = _to_gpu(layer, blocks)

_shared_layer_gauge(::Layer) = nothing

function _shared_gpu_gauge_for_layers(layers::Vector{<:Layer}, blocks)
    for layer in layers
        U = _shared_layer_gauge(layer)
        U === nothing && continue
        return _to_gpu_gauge(U; blocks=blocks)
    end
    return nothing
end

"""
    refresh_model!(layers, D_new, U_new)

Refresh stale CuArray GPU contexts after loading a model from JLD2.
- Ritz_Layer, PtcRayleighQuotient_Layer: refresh D
- PTLayer: refresh U
- fermionfield_Layer: refresh x
- GramSchmidt_Layer: refresh x
- LossSumLayer: recurse into sublayers
"""
function refresh_model!(layers::Vector{<:Layer}, D_new, U_new)
    for layer in layers
        _refresh_layer!(layer, D_new, U_new)
    end
    return layers
end

_refresh_layer!(layer::Layer, D_new, U_new) = nothing

_refresh_forward_error!(layer::Layer, D_new) = nothing

function refresh_forward_errors!(layers::Vector{<:Layer}, D_new)
    for layer in layers
        _refresh_forward_error!(layer, D_new)
    end
    return layers
end

function Base.:+(a::A, b::B) where {A<:Layer, B<:Layer}
    return LossSumLayer(a, b)
end

function (layers::Vector{T})(x::Any) where T<:Layer
    result = x
    for layer in layers
        result = layer(result)
    end
    return result
end

# Gaugefields substitute_U!
# Staging buffers avoid extra CUDA mallocs and GPU synchronization.
# convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
#   = 1 + mod(div(b-1,blocks_s),blocks) + mod(div(r-1,blocknumbers_s),blocknumbers)*blocks
@inline function _cvt_br(b, r, blk, blk_s, nblk, nblk_s)
    return 1 + mod(div(b - 1, blk_s), blk) + mod(div(r - 1, nblk_s), nblk) * blk
end

function _clear_forward_caches!(layers, first_layer::Int=1, last_layer::Int=length(layers)-1)
    for i in max(first_layer, 1):min(last_layer, length(layers)-1)
        _clear_layer_forward_cache!(layers[i])
    end
    return nothing
end

function _force_cuda_reclaim!()
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    CUDA.synchronize()
    return nothing
end

function _accum_gradient!(d::Vector{Any}, j::Int, gi)
    gi !== nothing && (d[j] = d[j] === nothing ? gi : d[j] + gi)
    return nothing
end
