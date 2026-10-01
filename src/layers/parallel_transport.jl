# PTLayer: transports each input field along its path and mixes the results
# with spin-matrix weights, y_j = sum_i W_ji T_i x_i, where T_i is the parallel
# transport along path i.

mutable struct PTLayer{G<:AbstractGaugefields, T<:AbstractSpinMatrix} <: Layer
    paths::Vector{Paths}
    U::Vector{G}
    Out::Int
    W::Vector{T}
    activation::ActivationFunction
    # --- 캐시 버퍼 (매 epoch deepcopy 방지) ---
    _hopped::Vector{WF3x1}      # forward에서 hop 결과 저장 (gradient에서 재사용)
    _result::Vector{WF3x1}      # forward 출력 버퍼
    _initialized::Bool           # 버퍼 초기화 여부
    # --- DeltaRule 캐시 버퍼 ---
    _invpaths::Vector{Paths}     # reverse_paths 캐시 (paths 불변이므로 1회만 계산)
    _d_conj_buf::Vector{WF3x1}  # conj(d) 작업용 버퍼
    _ci_buffers::Vector{WF3x1}  # 스레드별 ci 버퍼
    _delta_init::Bool            # DeltaRule 버퍼 초기화 여부
    _Wadj::Vector{T}             # adjoint(W) 캐시 벡터
    _invhops::Vector{Any}        # 스레드별 Hopping_operator 캐시
    _hop_out::Vector{Vector{WF3x1CPU}} # 스레드별 hop 출력 버퍼 (1-element, CPU 전용)
    # --- GPU-resident batched buffers ---
    _W_gpu::Any                 # CuArray{ComplexF64,3} (4,4,NW)
    _Wadj_gpu::Any              # CuArray{ComplexF64,3} (4,4,NW) adjoint(W)
    _hopped_all::Any            # CuArray{ComplexF64,5} (NC,NG,bs,rs,npaths) — forward hop cache for gradient
    _grad_gpu::Any              # CuArray{ComplexF64,3} (4,4,NW) gradient on GPU
    _adam_m_gpu::Any            # CuArray{ComplexF64,3} Adam 1st moment
    _adam_v_gpu::Any            # CuArray{ComplexF64,3} Adam 2nd moment
    # --- Batched hopping cache ---
    _hop_dirs_gpu::Any           # CuArray{Int8,2} (max_steps, npaths) direction table
    _hop_nsteps_gpu::Any         # CuArray{Int32,1} (npaths,) step counts
    _hop_max_steps::Int          # max steps across all paths
    _inv_hop_dirs_gpu::Any       # CuArray{Int8,2} (max_steps, npaths) inverse direction table
    _W_cpu_buf::Any              # Array{ComplexF64,3} (4,4,NW) — sync_W_from_gpu! staging

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

function _get_blocks(layer::PTLayer)
    if eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        bi = layer.U[1].blockinfo
        return bi.blocks
    end
    return nothing
end

_has_gpu(layer::PTLayer) = eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator

# --- PTLayer ---
function _to_cpu(layer::PTLayer)
    U_cpu = _to_cpu_gauge(layer.U)
    new_layer = PTLayer(layer.paths, U_cpu, layer.Out, layer.activation)
    W_cpu = _to_cpu_W(layer.W)
    new_layer.W = W_cpu
    return new_layer
end

_shared_layer_gauge(layer::PTLayer) = layer.U

# --- PTLayer ---
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

# function (layer::DenseLinearLayer)(x::WF3x1) 
#     Out = layer.Out
#     W   = layer.W

#     result = Vector{WF3x1}(undef, Out)
#     yi = deepcopy(x)   

    
#     @inbounds for i in 1:Out
# # yi 1
# mul!(yi, W[i], x) # yi = W[i]*x " "
# result[i] = layer.activation(yi) # activation alloc
#     end

#     return result
# end



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
    # CPU path keeps hopped fields locally.
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

    # 2) Apply each hopping path.
    if x[1] isa WF3x1CPU
        hop = Hopping_operator(layer.U, paths[1])
        hop_buf = WF3x1CPU[hopped[1]] # 1-element buffer
        @inbounds for i in 1:npaths
            hop.P = paths[i]
            hop_buf[1] = hopped[i]
            hop(x[i], hop_buf)
        end

    # 3) CPU: combine paths into Out outputs.
        @threads for j in 1:Out
            rj = result[j]
            @inbounds for i in 1:npaths
                w = W[(i-1)*Out + j]
                addmul!(rj, w, hopped[i])
            end
            result[j] = rj
        end
    else
    # GPU batched hop: reduce many hop kernels to max_steps kernels.
        NC = 3; NG = 4
        blocksize = x[1].blockinfo.blocksize
        rsize     = x[1].blockinfo.rsize

        # Lazy init GPU buffers; hopped_all uses a shared pool to avoid per-layer 628 MB.
        NW = npaths * Out
        if layer._W_gpu === nothing
            layer._W_gpu     = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._Wadj_gpu  = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._W_cpu_buf = Array{ComplexF64}(undef, NG, NG, NW)
            sync_W_to_gpu!(layer._W_gpu, W)
        end
        # Shared hopped_all buffer avoids 9 PTLayer * 628 MB = 5.6 GB.
        _result_buf  = get_shared_buf!(:result_all, NC, NG, blocksize, rsize, Out)
        _hop_temp    = get_shared_buf!(:hop_temp,   NC, NG, blocksize, rsize, npaths)
        _hopped_all  = get_shared_buf!(:hopped_all, NC, NG, blocksize, rsize, npaths)

        # lazy init batched hop tables
        if layer._hop_dirs_gpu === nothing
            max_steps = 0
            for i in 1:npaths
                steps_i = paths[i].various[1].steps
                ns = length(steps_i)
                # Identity path: single step with direction 0 means zero effective steps.
                if ns == 1 && steps_i[1].f == 0
                    ns = 0
                end
                max_steps = max(max_steps, ns)
            end
            layer._hop_max_steps = max_steps

            dirs_cpu = zeros(Int8, max_steps, npaths)
            nsteps_cpu = zeros(Int32, npaths)
            for i in 1:npaths
                steps_i = paths[i].various[1].steps
                ns = length(steps_i)
                if ns == 1 && steps_i[1].f == 0
                    nsteps_cpu[i] = Int32(0)
                else
                    nsteps_cpu[i] = Int32(ns)
                    for s in 1:ns
                        dirs_cpu[s, i] = Int8(steps_i[s].f)
                    end
                end
            end
            layer._hop_dirs_gpu   = CuArray(dirs_cpu)
            layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
        end

        # Batched hop: identity paths only pack; hop paths use batched_hopping_steps_gpu!.
        if layer._hop_max_steps > 0
            batched_hopping_steps_gpu!(
                _hopped_all, _hop_temp, x,
                layer.U, layer._hop_dirs_gpu, layer._hop_nsteps_gpu,
                layer._hop_max_steps, npaths, blocksize, rsize)
        else
            pack_fermions!(_hopped_all, x)
        end

        # Gradient checkpointing: do not store per-layer hop buffers.
        # Backward is sequential, so reuse shared pool(:hopped_all).
        layer._hopped_all = nothing


        # One batched forward kernel.
        batched_forward!(_result_buf, layer._W_gpu, _hopped_all, blocksize, rsize, npaths, Out)

        # Unpack result_all into result.
        unpack_fermions!(result, _result_buf)
    end

    return result
end

_refresh_layer!(layer::PTLayer, D_new, U_new) = (layer.U = U_new)

function gradient(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    Out=layer.Out
    W=layer.W
    paths=layer.paths

    n=length(layer.W)
    npaths = length(paths)

    @assert length(z) == npaths "PTLayer gradient: z의 길이는 paths 길이와 같아야 합니다. length(z)=$(length(z)), length(paths)=$(npaths)"
    @assert length(d) == Out "PTLayer gradient: d의 길이는 Out과 같아야 합니다. length(d)=$(length(d)), Out=$Out"
    @assert length(W) == npaths*Out "The length of W must be equal to length(paths)*Out."

    # gradient checkpointing: hopped_all은 per-layer로 저장하지 않고 z에서 재계산
    # (9 PTLayer × 628MB = 5.6GB 절약 — backward는 sequential이므로 shared buffer 안전)
    if layer._W_gpu !== nothing
        if d[1] isa WF3x1GPU
            # GPU batched: 625개 kernel → 1회 launch, GPU→CPU 전송 없음
            blocksize = d[1].blockinfo.blocksize
            rsize     = d[1].blockinfo.rsize
            NC = 3; NG = 4

            # forward에서 캐싱된 hop 결과 재사용 (hop 재계산 방지)
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

            # d_buf: 전역 공유 버퍼 사용
            _d_buf_g = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Out)
            if layer._grad_gpu === nothing || size(layer._grad_gpu, 3) != n
                layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, n)
            end

            pack_fermions!(_d_buf_g, d)
            batched_gradient!(layer._grad_gpu, _d_buf_g, _hopped_g,
                              blocksize, rsize, Out, npaths)

            return [layer._grad_gpu]
        elseif length(layer._hopped) == npaths
            # CPU: @threads 병렬화 (CPU path는 _hopped 캐시 그대로 사용)
            grad = Vector{typeof(W[1])}(undef, n)
            @threads for idx in 1:n
                j = ((idx-1) % Out) + 1
                i = ((idx-1) ÷ Out) + 1
                grad[idx]=Matmul!(d[j], layer._hopped[i])
            end
            return [grad]
        end
    else
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

function DeltaRule(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)
    result = Vector{WF3x1}(undef, Nz)
    for i in 1:Nz
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end

    # ① d_conj 버퍼 캐시 — CPU path에서만 필요
    is_gpu = z[1] isa WF3x1GPU && layer._W_gpu !== nothing

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
        @inbounds for j in 1:Nd
            conj_fermion!(d_conj[j], d[j])
        end
    elseif !layer._delta_init
        layer._delta_init = true
    end

    # ② 역경로 캐시 (paths 불변이므로 1회만 계산)
    if isempty(layer._invpaths) || length(layer._invpaths) != Nz
        layer._invpaths = Vector{typeof(paths[1])}(undef, Nz)
        @inbounds for i in 1:Nz
            layer._invpaths[i] = reverse_paths(paths[i])
        end
    end
    invpaths = layer._invpaths

    # ③ Wadj 캐시 — CPU path에서만 필요
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

    # ④ invhops + hop 출력 버퍼 캐시 (CPU path에서만 필요)
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

    # ⑤ 최적화: 먼저 c_i = Σ_j Wadj_{ij} * d_conj[j] 계산 후, hop 1회만 수행
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
            invhop(ci, hop_out[tid])  # preallocated 버전 사용
            conj_fermion!(result[i], hop_out[tid][1])
        end
    else
        # GPU batched: Wadj*conj(d) 를 1회 커널로
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        if layer._W_gpu !== nothing
            sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)
        else
            Wadj_cpu = Array{ComplexF64}(undef, NG, NG, NW)
            @inbounds for k in 1:NW
                sm = layer._Wadj[k].sm
                for jj in 1:4, ii in 1:4
                    Wadj_cpu[ii, jj, k] = sm[ii, jj]
                end
            end
            copyto!(layer._Wadj_gpu, Wadj_cpu)
        end

        # d_buf, ci_all: 공유 임시 버퍼 사용 (layers 간 재사용)
        _d_buf = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Nd)
        pack_fermions!(_d_buf, d)

        _ci_buf = get_shared_buf!(:ci_all, NC, NG, blocksize, rsize, Nz)

        # 1회 batched kernel: ci_all = Σ_j Wadj * conj(d)
        batched_delta!(_ci_buf, layer._Wadj_gpu, _d_buf, blocksize, rsize, Nd, Nz)

        # lazy init inverse direction table (forward init이 안 된 경우 여기서 수행)
        if layer._inv_hop_dirs_gpu === nothing
            max_steps = layer._hop_max_steps
            if max_steps == 0 && layer._hop_dirs_gpu === nothing
                max_steps = 0
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0; ns = 0; end
                    max_steps = max(max_steps, ns)
                end
                layer._hop_max_steps = max_steps

                dirs_cpu = zeros(Int8, max_steps, Nz)
                nsteps_cpu = zeros(Int32, Nz)
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        nsteps_cpu[i] = Int32(0)
                    else
                        nsteps_cpu[i] = Int32(ns)
                        for s in 1:ns; dirs_cpu[s, i] = Int8(steps_i[s].f); end
                    end
                end
                layer._hop_dirs_gpu   = CuArray(dirs_cpu)
                layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
            end

            max_steps = layer._hop_max_steps
            inv_dirs_cpu = zeros(Int8, max_steps, Nz)
            for i in 1:Nz
                inv_steps = invpaths[i].various[1].steps
                ns = length(inv_steps)
                if !(ns == 1 && inv_steps[1].f == 0)
                    for s in 1:ns; inv_dirs_cpu[s, i] = Int8(inv_steps[s].f); end
                end
            end
            layer._inv_hop_dirs_gpu = CuArray(inv_dirs_cpu)
        end

        # batched inverse hop: ci_buf ↔ shared hop_temp
        max_steps = layer._hop_max_steps
        if max_steps > 0
            _hop_temp = get_shared_buf!(:hop_temp, NC, NG, blocksize, rsize, Nz)
            bi = z[1].blockinfo
            NXl = z[1].NX; NYl = z[1].NY; NZl = z[1].NZ; NTl = z[1].NT
            batched_inv_hopping_gpu!(
                _ci_buf, _hop_temp, layer.U,
                layer._inv_hop_dirs_gpu, layer._hop_nsteps_gpu,
                max_steps, Nz, blocksize, rsize, bi, NXl, NYl, NZl, NTl)
        end

        # batch conj
        inplace_conj!(_ci_buf)

        # unpack → result
        unpack_fermions!(result, _ci_buf)
    end

    return result
end

function DeltaRule!(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."
    @assert length(delta) == length(z) "The length of delta must be equal to the length of z."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)

    # ① d_conj 버퍼 캐시 — CPU path에서만 필요
    is_gpu = z[1] isa WF3x1GPU && layer._W_gpu !== nothing

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
        # GPU path: ci_buffers, invhops, hop_out 불필요 (batched 처리)
        layer._delta_init = true
    end

    # ② 역경로 캐시 (paths 불변이므로 1회만 계산)
    if isempty(layer._invpaths) || length(layer._invpaths) != Nz
        layer._invpaths = Vector{typeof(paths[1])}(undef, Nz)
        @inbounds for i in 1:Nz
            layer._invpaths[i] = reverse_paths(paths[i])
        end
    end
    invpaths = layer._invpaths

    # ③ Wadj 캐시 — CPU path에서만 필요
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

    # ④ invhops + hop 출력 버퍼 캐시 (CPU path에서만 필요)
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

    # ⑤ 최적화: 먼저 c_i = Σ_j Wadj_{ij} * d_conj[j] 계산 후, hop 1회만 수행
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
            invhop(ci, hop_out[tid])  # preallocated 버전 사용
            conj_fermion!(delta[i], hop_out[tid][1])
        end
    else
        # GPU batched: Wadj*conj(d) 를 1회 커널로 전부 계산
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        # Wadj_gpu 동기화 (W_gpu가 있으면 GPU에서 직접, 없으면 CPU→GPU)
        if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        if layer._W_gpu !== nothing
            sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)
        else
            # fallback: CPU Wadj → GPU
            Wadj_cpu = Array{ComplexF64}(undef, NG, NG, NW)
            @inbounds for k in 1:NW
                sm = Wadj[k].sm
                for jj in 1:4, ii in 1:4
                    Wadj_cpu[ii, jj, k] = sm[ii, jj]
                end
            end
            copyto!(layer._Wadj_gpu, Wadj_cpu)
        end

        # d_buf, ci_all: 공유 임시 버퍼 사용 (layers 간 재사용)
        _d_buf = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Nd)
        pack_fermions!(_d_buf, d)

        _ci_buf = get_shared_buf!(:ci_all, NC, NG, blocksize, rsize, Nz)

        # 1회 batched kernel: ci_all = Σ_j Wadj * conj(d)
        batched_delta!(_ci_buf, layer._Wadj_gpu, _d_buf, blocksize, rsize, Nd, Nz)

        # lazy init inverse direction table for batched backward hop
        if layer._inv_hop_dirs_gpu === nothing
            max_steps = layer._hop_max_steps
            if max_steps == 0 && layer._hop_dirs_gpu === nothing
                # forward batched hop이 아직 init되지 않은 경우 — 여기서 직접 계산
                max_steps = 0
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        ns = 0
                    end
                    max_steps = max(max_steps, ns)
                end
                layer._hop_max_steps = max_steps

                # forward dirs + nsteps (forward에서 아직 init 안된 경우)
                dirs_cpu = zeros(Int8, max_steps, Nz)
                nsteps_cpu = zeros(Int32, Nz)
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        nsteps_cpu[i] = Int32(0)
                    else
                        nsteps_cpu[i] = Int32(ns)
                        for s in 1:ns
                            dirs_cpu[s, i] = Int8(steps_i[s].f)
                        end
                    end
                end
                layer._hop_dirs_gpu   = CuArray(dirs_cpu)
                layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
            end

            max_steps = layer._hop_max_steps
            # inverse dirs: reverse order & negate
            inv_dirs_cpu = zeros(Int8, max_steps, Nz)
            for i in 1:Nz
                inv_steps = invpaths[i].various[1].steps
                ns = length(inv_steps)
                if ns == 1 && inv_steps[1].f == 0
                    # identity — nsteps=0
                else
                    for s in 1:ns
                        inv_dirs_cpu[s, i] = Int8(inv_steps[s].f)
                    end
                end
            end
            layer._inv_hop_dirs_gpu = CuArray(inv_dirs_cpu)
        end

        # batched inverse hop: ci_buf ↔ shared hop_temp
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

        # batch conj
        inplace_conj!(_ci_buf)

        # unpack → delta
        unpack_fermions!(delta, _ci_buf)
    end

    return nothing
end

function update!(layer::PTLayer, opt::Adam, grad)
    # GPU batched path: grad[1]이 CuArray이면 GPU Adam
    if grad[1] isa CuArray
        NW = length(layer.W)
        NG = 4
        # lazy init Adam states on GPU
        if layer._adam_m_gpu === nothing || size(layer._adam_m_gpu, 3) != NW
            layer._adam_m_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._adam_v_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        gpu_adam_update!(layer._W_gpu, layer._adam_m_gpu, layer._adam_v_gpu, grad[1],
                         opt.α, opt.β1, opt.β2, opt.ε, opt.t)
    else
        # CPU fallback
        param = layer.W
        for i in 1:length(param)
            complexProcess(param[i], opt, grad[1][i])
        end
    end
end

# ──── Batched forward kernel ────
# result[c,g,b,r,j] = Σ_i Σ_g2  W[g,g2,(i-1)*Out+j] * hopped[c,g2,b,r,i]

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

625개 gpuaddmul! 대신 1회 커널 launch.
"""
function batched_forward!(result_all, W_gpu, hopped_all, blocksize, rsize, npaths, Out)
    NC = 3; NG = 4
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_forward!(
        result_all, W_gpu, hopped_all, NC, NG, npaths, Out)
end

# ──── Batched gradient kernel (inner-product reduction) ────
# M_all[fg,xg,widx] = Σ_{c,b,r} d[c,fg,b,r,j] * hopped[c,xg,b,r,i]
#  where  j = (widx-1)%Out + 1,  i = (widx-1)÷Out + 1

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

625개 kernel_matmul_batched! 대신 1회 launch.
결과는 grad_gpu (CuArray 4×4×NW) 에 직접 기록 → GPU→CPU 전송 불필요.
"""
function batched_gradient!(grad_gpu, d_all, hopped_all, blocksize, rsize, Out, npaths)
    NG = 4; NC = 3
    NW = npaths * Out
    threads = 256
    nblocks = NG * NG * NW          # 16 * NW
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_batched_gradient!(
        grad_gpu, d_all, hopped_all, NC, NG, blocksize, rsize, Out, npaths)
end

# ──── Batched DeltaRule kernel ────
# ci[c,g,b,r,i] = Σ_j Σ_g2  Wadj[g,g2,(i-1)*Out+j] * conj(d[c,g2,b,r,j])

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

DeltaRule의 gpuaddmul! 625회 → 1회 커널.
d_all은 원본 d (conj는 커널 내부에서 수행).
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

"""W_gpu → Wadj_gpu (transpose-conjugate, GPU 위에서)"""
function sync_Wadj_gpu!(Wadj_gpu, W_gpu)
    n = size(W_gpu, 3)
    threads = min(256, n)
    nblocks = cld(n, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adjoint_W!(Wadj_gpu, W_gpu, n)
end
