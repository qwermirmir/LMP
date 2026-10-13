# Type aliases and abstract types shared by the whole package.

# Wilson fermion fields with three colours, as created by
# Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true). On the CPU the
# data array is f[colour, x, y, z, t, spin]; on the GPU it is stored in blocks
# as f[colour, spin, site in block, block].
const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}
const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

# The four SU(3) link fields of a gauge configuration on the GPU.
const GPUgauge = Vector{Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, :cuda, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}}

# A 4x4 complex matrix acting on the spinor index; see core/spin_matrix.jl.
abstract type AbstractSpinMatrix <: AbstractArray{ComplexF64,2} end

# Every layer is a subtype of Layer. A layer `l` is applied to a vector of
# fields as `l(x)` and provides DeltaRule!/DeltaRule (backward pass),
# gradient (weight gradient) and update! (Adam step).
abstract type Layer end

# Applied to the output of a layer. Only Identity is in use.
abstract type ActivationFunction end

struct Identity <: ActivationFunction end

(activation::Identity)(x) = x

abstract type Operator end
