# DenseLinearLayer: mixes In input fields into Out output fields with
# spin-matrix weights, without any parallel transport.

"""
    DenseLinearLayer(In, Out, cuda, activation=Identity())

Linear layer on fermion fields with `In` inputs and `Out` outputs:

    y[i] = sum_k W[(k-1)*Out + i] * x[k],    i = 1..Out

Each `W[n]` is a 4x4 spin matrix applied at every site. `cuda` selects GPU
(`Spin_matrix_GPU`) or CPU (`Spin_matrix`) weights.

Backward pass, for output deltas `d[1:Out]`:

    delta[k] = sum_i conj(W[(k-1)*Out + i]^dagger conj(d[i]))

Weight gradient: `dL/dW[(k-1)*Out + i] = Matmul!(d[i], x[k])`, the 4x4 matrix
`sum_x d[i][a](x) x[k][b](x)`.

On the GPU it uses the batched kernels in `core/spin_mixing.jl` with the In
inputs in place of paths.
"""
mutable struct DenseLinearLayer{T<:AbstractSpinMatrix} <: Layer
    W::Vector{T}              # W[(k-1)*Out + i]: weight from input k to output i
    In::Int
    Out::Int
    cuda::Bool
    activation::ActivationFunction
    # CPU buffers
    _result::Vector{WF3x1}    # output fields
    _yi::Vector{WF3x1}        # 1-element work field
    _dl_init::Bool
    _conj_buf::Vector{WF3x1}  # [conj(d[i]), conj(y)] for DeltaRule!
    _delta_init_dl::Bool
    # GPU arrays
    _W_gpu::Any               # CuArray{ComplexF64,3} (4, 4, NW)
    _Wadj_gpu::Any            # CuArray{ComplexF64,3} (4, 4, NW), W^dagger
    _input_all::Any           # CuArray{ComplexF64,5} (NC, NG, bs, rs, In)
    _result_all::Any          # CuArray{ComplexF64,5} (NC, NG, bs, rs, Out)
    _d_buf::Any               # CuArray{ComplexF64,5} (NC, NG, bs, rs, Out)
    _ci_all::Any              # CuArray{ComplexF64,5} (NC, NG, bs, rs, In)
    _grad_gpu::Any            # CuArray{ComplexF64,3} (4, 4, NW)
    _adam_m_gpu::Any
    _adam_v_gpu::Any
    _W_cpu_buf::Any           # Array{ComplexF64,3} (4, 4, NW), staging for sync_W_from_gpu!

    function DenseLinearLayer(In::Int, Out::Int, cuda::Bool, activation::ActivationFunction=Identity() )
        if cuda == true
            W = Vector{Spin_matrix_GPU}(undef, In*Out)
            for i in 1:In*Out
                W[i] = Random_SpinM_GPU("complex")
            end
        else
            W = Vector{Spin_matrix}(undef, In*Out)
            for i in 1:In*Out
                W[i] = Random_SpinM("complex")
            end
        end
        return new{eltype(W)}(W, In, Out, cuda, activation, WF3x1[], WF3x1[], false, WF3x1[], false,
                              nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

function (layer::DenseLinearLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out = layer.Out
    In  = layer.In
    W   = layer.W
    NW  = length(W)

    @assert NW == In*Out "W length must be In*Out."

    # Initialize reusable outputs once.
    if !layer._dl_init || length(layer._result) != Out
        layer._result = Vector{WF3x1}(undef, Out)
        for i in 1:Out
            layer._result[i] = _alloc_result_wf(x[1])
        end
        layer._yi = [deepcopy(x[1])]
        layer._dl_init = true
    end

    result = layer._result

    if x[1] isa WF3x1GPU
        NC = 3; NG = 4
        blocksize = x[1].blockinfo.blocksize
        rsize     = x[1].blockinfo.rsize

        if layer._W_gpu === nothing
            layer._W_gpu      = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._W_cpu_buf  = Array{ComplexF64}(undef, NG, NG, NW)
            sync_W_to_gpu!(layer._W_gpu, W)
        end
        if layer._input_all === nothing || size(layer._input_all, 5) != In
            layer._input_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, In)
        end
        if layer._result_all === nothing || size(layer._result_all, 5) != Out
            layer._result_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, Out)
        end

        pack_fermions!(layer._input_all, x)
        batched_forward!(layer._result_all, layer._W_gpu, layer._input_all, blocksize, rsize, In, Out)
        unpack_fermions!(result, layer._result_all)
    else
        yi = layer._yi[1]
        @inbounds for j in 1:Out
            zero_wf!(result[j])
        end
        for k in 1:In
            @inbounds for i in 1:Out
                mul!(yi, W[i+(k-1)*Out], x[k])
                axpby!(1, yi, 1, result[i])
            end
        end
    end
    return result
end

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

function DeltaRule!(layer::DenseLinearLayer, ::Identity, z::Vector{T}, d::Vector{<:WF3x1}, delta::Vector{<:WF3x1}) where T<:WF3x1

    W   = layer.W
    In  = layer.In
    Out = layer.Out
    NW  = length(W)

    if z[1] isa WF3x1GPU && layer._W_gpu !== nothing
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        if layer._Wadj_gpu === nothing
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)

        if layer._d_buf === nothing || size(layer._d_buf, 5) != Out
            layer._d_buf = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, Out)
        end
        pack_fermions!(layer._d_buf, d)

        if layer._ci_all === nothing || size(layer._ci_all, 5) != In
            layer._ci_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, In)
        end

        batched_delta!(layer._ci_all, layer._Wadj_gpu, layer._d_buf, blocksize, rsize, Out, In)
        unpack_fermions!(delta, layer._ci_all)
        # batched_delta! computes W^dagger conj(d); the outer conj matches the CPU path.
        for k in 1:In
            conj_fermion!(delta[k], delta[k])
        end
    else
        if !layer._delta_init_dl || typeof(layer._conj_buf[1]) != typeof(z[1])
            layer._conj_buf = [deepcopy(z[1]), deepcopy(z[1])]  # [conj_d, conj_y]
            layer._delta_init_dl = true
        end
        conj_d = layer._conj_buf[1]
        conj_y = layer._conj_buf[2]
        y = layer._yi[1]

        zero_wf!(delta)
        for k in 1:In
            for i in 1:Out
                conj_fermion!(conj_d, d[i])
                mul!(y, adjoint(W[i+(k-1)*Out]), conj_d)
                conj_fermion!(conj_y, y)
                axpby!(1, conj_y, 1, delta[k])
            end
        end
    end
    return nothing
end

# Allocating version of DeltaRule!. It does not rely on the forward-pass buffers.
function DeltaRule(layer::DenseLinearLayer, ::Identity, z::Vector{T}, d::Vector{WF3x1} ) where T<:WF3x1

    W=layer.W

    result=Vector{WF3x1}(undef,length(z))
    y=similar(z[1])
    for i in 1:length(z)
        result[i]=deepcopy(y)
    end

    for k in 1:length(z)
        for i in 1:length(d)
            mul!(y,adjoint(W[i+(k-1)*length(d)]),(conj(d[i])))
            axpby!(1,conj(y),1,result[k])
        end
    end
    return result
end

# ---------------------------------------------------------------------------
# Weight gradient
# ---------------------------------------------------------------------------

# Returns [grad_gpu] on the GPU and a vector of spin matrices on the CPU.
function gradient(layer::DenseLinearLayer, ::Identity,
                  z::AbstractVector{TZ},
                  d::AbstractVector{TD},
                  M::Any) where {TZ<:WF3x1, TD<:WF3x1}

    n = length(layer.W)
    nd = length(d)
    nz = length(z)

    if d[1] isa WF3x1GPU && layer._W_gpu !== nothing
        NC = 3; NG = 4
        blocksize = d[1].blockinfo.blocksize
        rsize     = d[1].blockinfo.rsize

        if layer._grad_gpu === nothing
            layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, n)
        end
        # Pack d and z again: the buffers may hold other fields by now.
        if layer._d_buf === nothing || size(layer._d_buf, 5) != nd
            layer._d_buf = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, nd)
        end
        pack_fermions!(layer._d_buf, d)

        if layer._input_all === nothing || size(layer._input_all, 5) != nz
            layer._input_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, nz)
        end
        pack_fermions!(layer._input_all, z)

        batched_gradient!(layer._grad_gpu, layer._d_buf, layer._input_all,
                          blocksize, rsize, nd, nz)

        return [layer._grad_gpu]
    else
        grad = Vector{typeof(layer.W[1])}(undef, n)
        @threads for idx in 1:n
            i = ((idx-1) % nd) + 1
            k = ((idx-1) ÷ nd) + 1
            grad[idx] = Matmul!(d[i], z[k])
        end
        return grad
    end
end

# ---------------------------------------------------------------------------
# Adam update
# ---------------------------------------------------------------------------

function update!(layer::DenseLinearLayer, opt::Adam, grad)
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
        if length(param) == 1
            complexProcess(param, opt, grad[1])
        else
            for i in 1:length(param)
                complexProcess(param[i], opt, grad[i])
            end
        end
    end
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion
# ---------------------------------------------------------------------------

_has_gpu(layer::DenseLinearLayer{Spin_matrix_GPU}) = true

function _to_cpu(layer::DenseLinearLayer{Spin_matrix_GPU})
    new_layer = DenseLinearLayer(layer.In, layer.Out, false, layer.activation)
    W_cpu = Convertsm(layer.W)
    for j in eachindex(W_cpu)
        new_layer.W[j] = W_cpu[j]
    end
    return new_layer
end

function _to_gpu(layer::DenseLinearLayer{Spin_matrix}, blocks=nothing)
    new_layer = DenseLinearLayer(layer.In, layer.Out, true, layer.activation)
    W_gpu = [Spin_matrix_GPU(w) for w in layer.W]
    for j in eachindex(W_gpu)
        new_layer.W[j] = W_gpu[j]
    end
    return new_layer
end
