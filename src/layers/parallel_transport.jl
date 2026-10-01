# PTLayer: the parallel-transport layer. Each input field is transported along
# its own path and the transported fields are mixed by spin-matrix weights.

"""
    PTLayer(paths, U, Out, activation=Identity())

Parallel-transport layer with `length(paths)` inputs and `Out` outputs.

Forward pass, for input fields `x[1:npaths]`:

    y[j] = sum_i W[(i-1)*Out + j] * T_i x[i],    j = 1..Out

where `T_i` transports a field along `paths[i]` through the links `U` (see
`Hopping_operator`) and each `W[k]` is a 4x4 spin matrix acting on the spinor
index at every site. The layer is gauge covariant: transforming `U` and every
`x[i]` by a gauge transformation `g` transforms every `y[j]` by `g`.

Backward pass, for output deltas `d[1:Out]`:

    delta[i] = conj(T_i^-1 sum_j W[(i-1)*Out + j]^dagger conj(d[j]))

Weight gradient, summing over sites and colour:

    dL/dW[(i-1)*Out + j][a, b] = sum_x d[j][a](x) (T_i x[i])[b](x)

On the GPU the weights live in `_W_gpu` as a `4 x 4 x NW` array. The paths are
transported by `batched_hopping_steps_gpu!` and mixed by the kernels in
`core/spin_mixing.jl`.
"""
mutable struct PTLayer{G<:AbstractGaugefields, T<:AbstractSpinMatrix} <: Layer
    paths::Vector{Paths}
    U::Vector{G}
    Out::Int
    W::Vector{T}              # W[(i-1)*Out + j]: weight from path i to output j
    activation::ActivationFunction
    # Forward buffers
    _hopped::Vector{WF3x1}    # CPU: transported inputs T_i x[i], reused by gradient
    _result::Vector{WF3x1}    # output fields
    _initialized::Bool
    # Backward buffers (CPU)
    _invpaths::Vector{Paths}  # reverse_paths(paths[i]); paths never change
    _d_conj_buf::Vector{WF3x1}  # conj(d[j])
    _ci_buffers::Vector{WF3x1}  # one work field per thread
    _delta_init::Bool
    _Wadj::Vector{T}          # W[k]^dagger
    _invhops::Vector{Any}     # one Hopping_operator per thread
    _hop_out::Vector{Vector{WF3x1CPU}}  # one 1-element output buffer per thread
    # GPU arrays
    _W_gpu::Any               # CuArray{ComplexF64,3} (4, 4, NW)
    _Wadj_gpu::Any            # CuArray{ComplexF64,3} (4, 4, NW), W^dagger
    _hopped_all::Any          # unused: the transported inputs live in a shared buffer
    _grad_gpu::Any            # CuArray{ComplexF64,3} (4, 4, NW)
    _adam_m_gpu::Any          # Adam first moment, same shape as _W_gpu
    _adam_v_gpu::Any          # Adam second moment
    # Direction tables for the batched GPU hop kernels
    _hop_dirs_gpu::Any        # CuArray{Int8,2} (max_steps, npaths)
    _hop_nsteps_gpu::Any      # CuArray{Int32,1} (npaths,)
    _hop_max_steps::Int
    _inv_hop_dirs_gpu::Any    # CuArray{Int8,2} (max_steps, npaths), reversed paths
    _W_cpu_buf::Any           # Array{ComplexF64,3} (4, 4, NW), staging for sync_W_from_gpu!

    function PTLayer(paths::Vector{Paths}, U::Vector{G}, Out::Int,
                     activation::ActivationFunction=Identity()) where {G<:AbstractGaugefields}

        if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
            W = Vector{Spin_matrix_GPU}(undef, length(paths)*Out)
            for i in eachindex(W)
                W[i] = Random_SpinM_GPU("complex")
            end
            return new{G, Spin_matrix_GPU}(paths, U, Out, W, activation,
                       WF3x1[], WF3x1[], false,
                       Paths[], WF3x1[], WF3x1[], false,
                       Spin_matrix_GPU[], Any[], Vector{WF3x1CPU}[],
                       nothing, nothing, nothing, nothing, nothing, nothing,
                       nothing, nothing, 0, nothing, nothing)
        else
            W = Vector{Spin_matrix}(undef, length(paths)*Out)
            for i in eachindex(W)
                W[i] = Random_SpinM("complex")
            end
            return new{G, Spin_matrix}(paths, U, Out, W, activation,
                       WF3x1[], WF3x1[], false,
                       Paths[], WF3x1[], WF3x1[], false,
                       Spin_matrix[], Any[], Vector{WF3x1CPU}[],
                       nothing, nothing, nothing, nothing, nothing, nothing,
                       nothing, nothing, 0, nothing, nothing)
        end
    end
end

function Base.show(io::IO, layer::PTLayer)
    print(io, "PTLayer(paths=$(length(layer.paths)), Out=$(layer.Out))")
end

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

function (layer::PTLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    npaths = length(paths)

    @assert npaths == length(x) "x length must match paths."
    @assert length(W) == Out * npaths "W length must be Out*paths."

    # 1) Initialize reusable buffers.
    _is_gpu_pt = x[1] isa WF3x1GPU
    if !layer._initialized || length(layer._result) != Out
        if !_is_gpu_pt
            # The CPU path keeps the transported fields for the gradient.
            layer._hopped = Vector{WF3x1}(undef, npaths)
            for i in 1:npaths
                layer._hopped[i] = deepcopy(x[1])
            end
        end
        layer._result = Vector{WF3x1}(undef, Out)
        for j in 1:Out
            layer._result[j] = _alloc_result_wf(x[1])
        end
        layer._initialized = true
    end

    hopped = layer._hopped
    result = layer._result

    # Zero outputs before accumulation.
    @inbounds for j in 1:Out
        zero_wf!(result[j])
    end

    if x[1] isa WF3x1CPU
        # 2) CPU: transport each input along its path.
        hop = Hopping_operator(layer.U, paths[1])
        hop_buf = WF3x1CPU[hopped[1]] # 1-element buffer
        @inbounds for i in 1:npaths
            hop.P = paths[i]
            hop_buf[1] = hopped[i]
            hop(x[i], hop_buf)
        end

        # 3) CPU: mix the transported fields into the Out outputs.
        @threads for j in 1:Out
            rj = result[j]
            @inbounds for i in 1:npaths
                w = W[(i-1)*Out + j]
                addmul!(rj, w, hopped[i])
            end
            result[j] = rj
        end
    else
        # GPU: transport all paths with max_steps batched kernels, then mix with
        # one batched kernel.
        NC = 3; NG = 4
        blocksize = x[1].blockinfo.blocksize
        rsize     = x[1].blockinfo.rsize

        NW = npaths * Out
        if layer._W_gpu === nothing
            layer._W_gpu     = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._Wadj_gpu  = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._W_cpu_buf = Array{ComplexF64}(undef, NG, NG, NW)
            sync_W_to_gpu!(layer._W_gpu, W)
        end
        # The transported inputs go into a buffer shared by all PTLayers instead
        # of one per layer: with 9 layers on a 16^3x32 lattice that saves
        # 9 x 628 MB. The backward pass recomputes them (gradient checkpointing).
        _result_buf  = get_shared_buf!(:result_all, NC, NG, blocksize, rsize, Out)
        _hop_temp    = get_shared_buf!(:hop_temp,   NC, NG, blocksize, rsize, npaths)
        _hopped_all  = get_shared_buf!(:hopped_all, NC, NG, blocksize, rsize, npaths)

        if layer._hop_dirs_gpu === nothing
            _init_hop_tables!(layer)
        end

        # Identity paths only need packing; other paths are transported.
        if layer._hop_max_steps > 0
            batched_hopping_steps_gpu!(
                _hopped_all, _hop_temp, x,
                layer.U, layer._hop_dirs_gpu, layer._hop_nsteps_gpu,
                layer._hop_max_steps, npaths, blocksize, rsize)
        else
            pack_fermions!(_hopped_all, x)
        end
        layer._hopped_all = nothing

        batched_forward!(_result_buf, layer._W_gpu, _hopped_all, blocksize, rsize, npaths, Out)
        unpack_fermions!(result, _result_buf)
    end

    return result
end

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

"""
    DeltaRule!(layer::PTLayer, ::Identity, z, d, delta)

Writes the input deltas into `delta` given the layer inputs `z` and the output
deltas `d`. See [`PTLayer`](@ref) for the formula.
"""
function DeltaRule!(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."
    @assert length(delta) == length(z) "The length of delta must be equal to the length of z."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)

    is_gpu = z[1] isa WF3x1GPU && layer._W_gpu !== nothing

    # CPU: conj(d[j]) buffers.
    if !is_gpu
        if !layer._delta_init || length(layer._d_conj_buf) != Nd
            layer._d_conj_buf = Vector{WF3x1}(undef, Nd)
            for i in 1:Nd
                layer._d_conj_buf[i] = deepcopy(d[1])
            end
            layer._ci_buffers = [deepcopy(z[1]) for _ in 1:nthreads()]
            layer._delta_init = true
        end
        d_conj = layer._d_conj_buf
        @inbounds for i in 1:Nd
            conj_fermion!(d_conj[i], d[i])
        end
    elseif !layer._delta_init
        layer._delta_init = true
    end

    # Reversed paths, computed once.
    if isempty(layer._invpaths) || length(layer._invpaths) != Nz
        layer._invpaths = Vector{typeof(paths[1])}(undef, Nz)
        @inbounds for i in 1:Nz
            layer._invpaths[i] = reverse_paths(paths[i])
        end
    end
    invpaths = layer._invpaths

    # CPU: W^dagger, refreshed every call because W changes during training.
    if !is_gpu
        if length(layer._Wadj) != NW
            layer._Wadj = Vector{typeof(W[1])}(undef, NW)
            @inbounds for k in 1:NW
                layer._Wadj[k] = adjoint(W[k])
            end
        end
        Wadj = layer._Wadj
        @inbounds for k in 1:NW
            wsm = W[k].sm
            wadj_sm = Wadj[k].sm
            for j in 1:4
                for i in 1:4
                    wadj_sm[i,j] = conj(wsm[j,i])
                end
            end
        end
    end

    # CPU: one reverse-transport operator and output buffer per thread.
    if !is_gpu
        nt = nthreads()
        if isempty(layer._invhops) || length(layer._invhops) != nt
            layer._invhops = [Hopping_operator(layer.U, invpaths[1]) for _ in 1:nt]
            layer._hop_out = [WF3x1CPU[deepcopy(z[1])] for _ in 1:nt]
        else
            @inbounds for t in 1:nt
                layer._invhops[t].Gauge = layer.U
            end
        end
    end

    # Mix first, c_i = sum_j W_ij^dagger conj(d[j]), so each path is transported once.
    if z[1] isa WF3x1CPU
        ci_buffers = layer._ci_buffers
        invhops = layer._invhops
        hop_out = layer._hop_out

        @threads for i in 1:Nz
            tid = threadid()
            ci = ci_buffers[tid]
            zero_wf!(ci)
            base = (i-1) * Out
            @inbounds for j in 1:Out
                addmul!(ci, Wadj[base + j], d_conj[j])
            end
            invhop = invhops[tid]
            invhop.P = invpaths[i]
            invhop(ci, hop_out[tid])
            conj_fermion!(delta[i], hop_out[tid][1])
        end
    else
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        # W^dagger on the GPU, from _W_gpu when it exists, otherwise from the CPU copy.
        if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        if layer._W_gpu !== nothing
            sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)
        else
            Wadj_cpu = Array{ComplexF64}(undef, NG, NG, NW)
            @inbounds for k in 1:NW
                sm = Wadj[k].sm
                for jj in 1:4, ii in 1:4
                    Wadj_cpu[ii, jj, k] = sm[ii, jj]
                end
            end
            copyto!(layer._Wadj_gpu, Wadj_cpu)
        end

        # Work buffers are shared between layers.
        _d_buf = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Nd)
        pack_fermions!(_d_buf, d)

        _ci_buf = get_shared_buf!(:ci_all, NC, NG, blocksize, rsize, Nz)

        # c_i = sum_j W_ij^dagger conj(d[j]) in one kernel.
        batched_delta!(_ci_buf, layer._Wadj_gpu, _d_buf, blocksize, rsize, Nd, Nz)

        if layer._inv_hop_dirs_gpu === nothing
            # The forward pass normally builds the forward tables first.
            if layer._hop_max_steps == 0 && layer._hop_dirs_gpu === nothing
                _init_hop_tables!(layer)
            end
            inv_dirs_cpu, _, _ = _hop_direction_table(invpaths, layer._hop_max_steps)
            layer._inv_hop_dirs_gpu = CuArray(inv_dirs_cpu)
        end

        # Transport c_i back along the reversed path.
        max_steps = layer._hop_max_steps
        if max_steps > 0
            _hop_temp = get_shared_buf!(:hop_temp, NC, NG, blocksize, rsize, Nz)
            bi = z[1].blockinfo
            NX = z[1].NX; NY = z[1].NY; NZ_ = z[1].NZ; NT = z[1].NT
            batched_inv_hopping_gpu!(
                _ci_buf, _hop_temp, layer.U,
                layer._inv_hop_dirs_gpu, layer._hop_nsteps_gpu,
                max_steps, Nz, blocksize, rsize, bi, NX, NY, NZ_, NT)
        end

        inplace_conj!(_ci_buf)
        unpack_fermions!(delta, _ci_buf)
    end

    return nothing
end

"""
    DeltaRule(layer::PTLayer, ::Identity, z, d)

Allocating version of [`DeltaRule!`](@ref): returns the input deltas.
"""
function DeltaRule(layer::PTLayer, act::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1
    result = Vector{WF3x1}(undef, length(z))
    for i in eachindex(result)
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end
    DeltaRule!(layer, act, z, d, result)
    return result
end

# ---------------------------------------------------------------------------
# Weight gradient
# ---------------------------------------------------------------------------

"""
    gradient(layer::PTLayer, ::Identity, z, d, γ)

Returns `[grad]`, the gradient of the loss with respect to every weight, for
layer inputs `z` and output deltas `d`. On the GPU `grad` is the `4 x 4 x NW`
array `layer._grad_gpu`; on the CPU it is a vector of spin matrices.
"""
function gradient(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    Out=layer.Out
    W=layer.W
    paths=layer.paths

    n=length(layer.W)
    npaths = length(paths)

    @assert length(z) == npaths "PTLayer gradient: length(z) must equal the number of paths. length(z)=$(length(z)), length(paths)=$(npaths)"
    @assert length(d) == Out "PTLayer gradient: length(d) must equal Out. length(d)=$(length(d)), Out=$Out"
    @assert length(W) == npaths*Out "The length of W must be equal to length(paths)*Out."

    if layer._W_gpu !== nothing
        if d[1] isa WF3x1GPU
            # GPU: one kernel for all weights. The transported inputs are not
            # stored by the forward pass, so recompute them from z.
            blocksize = d[1].blockinfo.blocksize
            rsize     = d[1].blockinfo.rsize
            NC = 3; NG = 4

            if layer._hopped_all !== nothing && size(layer._hopped_all, 5) == npaths
                _hopped_g = layer._hopped_all
            else
                _hopped_g = get_shared_buf!(:hopped_all, NC, NG, blocksize, rsize, npaths)
                _hop_t    = get_shared_buf!(:hop_temp,   NC, NG, blocksize, rsize, npaths)
                pack_fermions!(_hopped_g, z)
                if layer._hop_max_steps > 0 && layer._hop_dirs_gpu !== nothing
                    batched_hopping_steps_gpu!(
                        _hopped_g, _hop_t, z,
                        layer.U, layer._hop_dirs_gpu, layer._hop_nsteps_gpu,
                        layer._hop_max_steps, npaths, blocksize, rsize)
                end
            end

            _d_buf_g = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Out)
            if layer._grad_gpu === nothing || size(layer._grad_gpu, 3) != n
                layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, n)
            end

            pack_fermions!(_d_buf_g, d)
            batched_gradient!(layer._grad_gpu, _d_buf_g, _hopped_g,
                              blocksize, rsize, Out, npaths)

            return [layer._grad_gpu]
        elseif length(layer._hopped) == npaths
            # CPU fields with the transported inputs cached by the forward pass.
            grad = Vector{typeof(W[1])}(undef, n)
            @threads for idx in 1:n
                j = ((idx-1) % Out) + 1
                i = ((idx-1) ÷ Out) + 1
                grad[idx]=Matmul!(d[j], layer._hopped[i])
            end
            return [grad]
        end
    else
        # Weights not on the GPU: transport each input again.
        grad = Vector{typeof(W[1])}(undef, n)
        hop=Hopping_operator(layer.U,paths[1])
        for i in 1:npaths
            hop.P=paths[i]
            u = hop(z[i])
            if d[1] isa WF3x1GPU
                Qs = Vector{WF3x1GPU}(undef, Out)
                Ks = Vector{WF3x1GPU}(undef, Out)
                @inbounds for j in 1:Out
                    Qs[j] = d[j]
                    Ks[j] = u[1]
                end
                batch_result = Matmul_batch(Qs, Ks)
                @inbounds for j in 1:Out
                    grad[(i-1)*Out+j] = batch_result[j]
                end
            else
                @threads for j in 1:Out
                    grad[(i-1)*Out+j]=Matmul!(d[j],(u[1]))
                end
            end
        end
        return [grad]
    end

end

# ---------------------------------------------------------------------------
# Adam update
# ---------------------------------------------------------------------------

function update!(layer::PTLayer, opt::Adam, grad)
    if grad[1] isa CuArray
        NW = length(layer.W)
        NG = 4
        if layer._adam_m_gpu === nothing || size(layer._adam_m_gpu, 3) != NW
            layer._adam_m_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._adam_v_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        gpu_adam_update!(layer._W_gpu, layer._adam_m_gpu, layer._adam_v_gpu, grad[1],
                         opt.α, opt.β1, opt.β2, opt.ε, opt.t)
    else
        param = layer.W
        for i in 1:length(param)
            complexProcess(param[i], opt, grad[1][i])
        end
    end
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion
# ---------------------------------------------------------------------------

_has_gpu(layer::PTLayer) = eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator

function _get_blocks(layer::PTLayer)
    if eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        bi = layer.U[1].blockinfo
        return bi.blocks
    end
    return nothing
end

_shared_layer_gauge(layer::PTLayer) = layer.U

function _to_cpu(layer::PTLayer)
    U_cpu = _to_cpu_gauge(layer.U)
    new_layer = PTLayer(layer.paths, U_cpu, layer.Out, layer.activation)
    W_cpu = _to_cpu_W(layer.W)
    new_layer.W = W_cpu
    return new_layer
end

function _to_gpu(layer::PTLayer, blocks=nothing)
    U_gpu = _to_gpu_gauge(layer.U; blocks=blocks)
    new_layer = PTLayer(layer.paths, U_gpu, layer.Out, layer.activation)
    W_gpu = _to_gpu_W(layer.W)
    new_layer.W = W_gpu
    return new_layer
end

function _to_gpu(layer::PTLayer, blocks, shared_U)
    U_gpu = shared_U === nothing ? _to_gpu_gauge(layer.U; blocks=blocks) : shared_U
    new_layer = PTLayer(layer.paths, U_gpu, layer.Out, layer.activation)
    W_gpu = _to_gpu_W(layer.W)
    new_layer.W = W_gpu
    return new_layer
end

_refresh_layer!(layer::PTLayer, D_new, U_new) = (layer.U = U_new)

# ---------------------------------------------------------------------------
# Direction tables for the batched GPU hop kernels
# ---------------------------------------------------------------------------

# Number of unit steps in a path; the identity path (a single step of
# direction 0) has none.
function _path_nsteps(path)
    steps = path.various[1].steps
    return length(steps) == 1 && steps[1].f == 0 ? 0 : length(steps)
end

# Direction table for the batched hop kernels: column i lists the step
# directions of paths[i], padded with zeros to max_steps rows.
function _hop_direction_table(paths, max_steps=maximum(_path_nsteps, paths; init=0))
    dirs = zeros(Int8, max_steps, length(paths))
    nsteps = zeros(Int32, length(paths))
    for (i, path) in enumerate(paths)
        ns = _path_nsteps(path)
        nsteps[i] = Int32(ns)
        for s in 1:ns
            dirs[s, i] = Int8(path.various[1].steps[s].f)
        end
    end
    return dirs, nsteps, max_steps
end

function _init_hop_tables!(layer::PTLayer)
    dirs_cpu, nsteps_cpu, max_steps = _hop_direction_table(layer.paths)
    layer._hop_max_steps  = max_steps
    layer._hop_dirs_gpu   = CuArray(dirs_cpu)
    layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
    return layer
end
