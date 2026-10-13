# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

mutable struct HoppingRayleighQuotient_Layer <: Layer
    hop::Hopping_operator
    x::WF3x1
    activation::ActivationFunction
    function HoppingRayleighQuotient_Layer(hop::Hopping_operator, x::WF3x1, activation::ActivationFunction= Identity())
        new(hop, x, activation)
    end
end

function _get_blocks(layer::HoppingRayleighQuotient_Layer)
    if eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return layer.hop.Gauge[1].blockinfo.blocks
    end
    return nothing
end

function _has_gpu(layer::HoppingRayleighQuotient_Layer)
    return eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator || layer.x isa WF3x1GPU
end

# --- HoppingRayleighQuotient_Layer ---
function _to_cpu(layer::HoppingRayleighQuotient_Layer)
    layer.hop = _to_cpu_hop(layer.hop)
    if layer.x isa WF3x1GPU
        layer.x = _to_cpu_fermion(layer.x)
    end
    return layer
end

_shared_layer_gauge(layer::HoppingRayleighQuotient_Layer) = layer.hop.Gauge

# --- HoppingRayleighQuotient_Layer ---
function _to_gpu(layer::HoppingRayleighQuotient_Layer, blocks=nothing)
    layer.hop = _to_gpu_hop(layer.hop; blocks=blocks)
    if layer.x isa WF3x1CPU
        layer.x = _to_gpu_fermion(layer.x; blocks=blocks)
    end
    return layer
end

function _to_gpu(layer::HoppingRayleighQuotient_Layer, blocks, shared_U)
    layer.hop = _to_gpu_hop(layer.hop; blocks=blocks, shared_U=shared_U)
    if layer.x isa WF3x1CPU
        layer.x = _to_gpu_fermion(layer.x; blocks=blocks)
    end
    return layer
end

function (layer::HoppingRayleighQuotient_Layer)(D::Dirac_operator)


    x=layer.x

    Hx=layer.hop(x)

    y=similar(x)
    for i in 1:length(Hx)
        axpby!(1,Hx[i],1,y)
    end
    u=D*y
    
    rq=inner_product(u,u)/inner_product(y,y)
    
    return rq
end

function gradient(layer::HoppingRayleighQuotient_Layer, ::Identity, Dirac::Dirac_operator)


    z=layer.hop(layer.x)


    x=similar(layer.x)
    y=similar(layer.x)

    
    for i in 1:length(z)
        axpby!(1,z[i],1,x)
    end 
    y=Dirac*x

    u=Dirac'*x
    invP = reverse_paths(layer.hop.P)
    invhop = Hopping_operator(layer.hop.Gauge, invP)
    phi=invhop(u)
    c=invhop(x)




    result=similar(y)
    for i in 1:length(z)
        result+=conj(phi[i])/norm2(x)-(norm2(y)/norm2(x)^2)*conj(c[i])
    end

    return [result]
end

# HoppingRayleighQuotient_Layer: x :: WF3x1
function update!(layer::HoppingRayleighQuotient_Layer, opt::Adam, grad::WF3x1)
    wf_adam_step!(layer.x, opt, grad)
    return nothing
end
