# Type aliases and abstract types shared by the whole package: the CPU and GPU
# Wilson fermion field types (WF3x1*), the Layer and activation-function
# abstractions, and the GPU gauge-field type.

# using LatticeQCD

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}

const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

abstract type AbstractSpinMatrix <: AbstractArray{ComplexF64,2} end

# using LatticeQCD




const GPUgauge=Vector{Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, :cuda, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}}

abstract type Layer end

abstract type ActivationFunction end

# export L2Norm_Layer

struct Identity <: ActivationFunction end

function (Identity)(x::WF3x1)
    return x
end # 항등함수

# Identity를 함수처럼 동작하게 만들기
function (activation::Identity)(x)
    return x  # 항등 함수
end

abstract type Operator end
