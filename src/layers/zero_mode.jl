# ZeroMode_Layer: loss that measures how close the input fields are to the null
# space of the Dirac operator.

mutable struct ZeroMode_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    _Dz::Vector{Any}
    _DdagDz::Vector{Any}
    _delta::Vector{Any}
    _losses::Vector{Float64}
    _denoms::Vector{Float64}
    _zeromode_init::Bool
    function ZeroMode_Layer(D::Any, activation::ActivationFunction=Identity(), Out=1)
        new(D, activation, Out, Any[], Any[], Any[], Float64[], Float64[], false)
    end
end

function Base.show(io::IO, layer::ZeroMode_Layer)
    print(io, "ZeroMode_Layer(Out=$(layer.Out))")
end

function _get_blocks(layer::ZeroMode_Layer)
    try
        U = layer.D.U
        if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
            return U[1].blockinfo.blocks
        end
    catch; end
    return nothing
end

function _has_gpu(layer::ZeroMode_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
end

function _to_cpu(layer::ZeroMode_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    layer._zeromode_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._delta = Any[]
    layer._losses = Float64[]
    layer._denoms = Float64[]
    return layer
end

function _shared_layer_gauge(layer::ZeroMode_Layer)
    try
        return layer.D.U
    catch
        return nothing
    end
end

function _to_gpu(layer::ZeroMode_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
    layer._zeromode_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._delta = Any[]
    layer._losses = Float64[]
    layer._denoms = Float64[]
    return layer
end

function _to_gpu(layer::ZeroMode_Layer, blocks, shared_U)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks, shared_U=shared_U)
    layer._zeromode_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._delta = Any[]
    layer._losses = Float64[]
    layer._denoms = Float64[]
    return layer
end

function _refresh_layer!(layer::ZeroMode_Layer, D_new, U_new)
    layer.D = D_new
    layer._zeromode_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._delta = Any[]
    layer._losses = Float64[]
    layer._denoms = Float64[]
    return nothing
end

function (layer::ZeroMode_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)

    need_reinit = !layer._zeromode_init || length(layer._Dz) != l
    if !need_reinit
        need_reinit = typeof(layer._Dz[1]) != typeof(x[1])
    end
    if need_reinit
        layer._Dz = Vector{Any}(undef, l)
        layer._DdagDz = Vector{Any}(undef, l)
        layer._losses = zeros(Float64, l)
        layer._denoms = zeros(Float64, l)
        for i in 1:l
            layer._Dz[i] = _alloc_result_wf(x[1])
            layer._DdagDz[i] = _alloc_result_wf(x[1])
        end
        layer._zeromode_init = true
    end

    Dx = layer._Dz
    DdagDx = layer._DdagDz
    if x[1] isa WF3x1GPU
        _kappa = _dirac_kappa(layer.D)
        _U = layer.D.U
        @inbounds for i in 1:l
            fused_wilson_mul_gpu!(Dx[i], x[i], _U, _kappa, false)
            fused_wilson_mul_gpu!(DdagDx[i], Dx[i], _U, _kappa, true)
        end
    else
        @inbounds for i in 1:l
            mul!(Dx[i], layer.D, x[i])
            mul!(DdagDx[i], layer.D', Dx[i])
        end
    end

    total = 0.0
    @inbounds for i in 1:l
        li = real(inner_product(Dx[i], Dx[i]))
        layer._losses[i] = li
        layer._denoms[i] = real(inner_product(x[i], x[i]))
        total += li
    end
    return total / l
end

function DeltaRule(layer::ZeroMode_Layer, ::Identity, z::AbstractVector{<:WF3x1}, batch::Int)
    result = similar(z)
    for i in eachindex(z)
        result[i] = similar(z[i])
    end
    return DeltaRule!(layer, layer.activation, z, result, batch)
end

function DeltaRule!(layer::ZeroMode_Layer, ::Identity, z::AbstractVector{<:WF3x1}, delta::AbstractVector{<:WF3x1}, batch::Int)
    l = length(z)
    length(layer._DdagDz) == l || error("ZeroMode_Layer forward must run before backward")
    for i in eachindex(delta)
        conj_fermion!(delta[i], layer._DdagDz[i])
        delta[i].f .= (2 / l) .* delta[i].f
    end
    return delta
end
