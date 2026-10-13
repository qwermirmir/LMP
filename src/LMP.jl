"""
    LMP

Machine-learning models built directly on lattice QCD fermion fields. The layers
are gauge covariant: they combine parallel transport along gauge links with
trainable 4x4 spin matrices, and the models are trained with Rayleigh-Ritz
objectives to learn low modes of the Wilson Dirac operator.

Source layout:

- `core/`: fermion fields, spin matrices, paths, parallel transport and the
  Dirac operator, on CPU and GPU.
- `layers/`: one file per layer type, holding its struct, forward pass,
  backward pass, weight gradient, CPU/GPU conversion and GPU kernels.
- `model/`: forward and backward passes through a list of layers, moving a
  model between CPU and GPU, and building the standard model.
- `training/`: the Adam optimizer, the training loop, checkpoints, gauge-file
  I/O and the training log.
- `solvers/`: DDalphaAMG and related multigrid solvers.
- `analysis/`: subspace comparison utilities.
"""
module LMP

__precompile__(false)

using Base.Threads
using Base.Iterators: take
using CUDA
using Gaugefields
using JLD2
using LatticeDiracOperators
using LinearAlgebra
using Random
using StaticArrays
using TimerOutputs
import PyPlot
using PyPlot: plt

# Functions from Base and the dependencies that LMP adds methods to.
import Base: getindex, setindex!, size, length, ndims, broadcastable
import Base.Broadcast: broadcast!
import LinearAlgebra: mul!
import Gaugefields: add_U!, clear_U!
import LatticeDiracOperators.Dirac_operators: substitute_fermion!, Shifted_fermionfields_4D_accelerator

# NVTX ranges mark GPU work for the Nsight profilers.
const NVTX = CUDA.NVTX
NVTX.activate()

include("core/types.jl")
include("core/spin_matrix.jl")
include("core/paths.jl")
include("core/gpu_lattice.jl")
include("core/gpu_buffers.jl")
include("core/fermion_ops.jl")
include("core/hopping.jl")
include("core/spin_mixing.jl")
include("core/dirac.jl")
include("core/clover.jl")
include("core/gauge_transform.jl")

# Layers add update! methods for Adam, so the optimizer comes first.
include("training/adam.jl")

include("layers/fermion_field.jl")
include("layers/dense_linear.jl")
include("layers/parallel_transport.jl")
include("layers/norm.jl")
include("layers/block_scalar.jl")
include("layers/ritz.jl")
include("layers/zero_mode.jl")
include("layers/orthogonal.jl")
include("layers/loss_sum.jl")

include("model/device.jl")
include("model/propagation.jl")

include("training/logging.jl")
include("training/gauge_io.jl")
include("training/checkpoint.jl")
include("training/train.jl")

include("solvers/ddalpha_amg.jl")
include("solvers/gdlambda_ddalpha_adapter.jl")
include("layers/coarse_correction_loss.jl")
include("solvers/gdlambda_amg.jl")
include("analysis/subspace_similarity.jl")

include("model/build.jl")
include("compat.jl")

using .DDalphaAMG

# Fields, spin matrices and paths
export Spin_matrix, Spin_matrix_GPU, Random_SpinM, Random_SpinM_GPU, Convertsm, Clifford, xavier_init!
export Paths, reverse_paths, Hopping_operator, gUgshift!
export inner_product, norm2, mean, addmul!, zero_wf!, Matmul!
export CloverDiracOperator, compute_Fmunu, refresh_clover!

# Layers
export Identity, Operator
export fermionfield_Layer, DenseLinearLayer, PTLayer, NormLayer, BlockScalar_Layer
export Ritz_Layer, ZeroMode_Layer, Orthogonal_Layer
export CoarseCorrectionLoss_Layer, coarse_correction_loss_forward!

# Models
export forward_propagation, forward_propagation!, backward_propagation!, gradient, DeltaRule
export substitute_Layer!, refresh_model!, refresh_fermionfield!, _to_cpu, _to_gpu, _to_cpu_gauge
export build_lmp_32pt_layers, load_lmp_model

# Training
export Opt, Adam, update!, advance_step!, sync_layers!, save_checkpoint, load_checkpoint!
export train

# Multigrid solvers
export DDalphaAMG_Setup, CoarseVector
export setup_DDalphaAMG, solve_DDalphaAMG, vcycle
export restrict_f2c, prolongate_c2f
export restrict_f2c!, prolongate_c2f!, coarse_apply!
export precompute_coarse_operator!
export mr_smooth!, sap_smooth!
export block_orthonormalize!, deflated_initial_guess
export gamma5!, chiral_project!
export ShiftedGamma5DiracOperator, Gamma5LeftShiftedDiracOperator
export shifted_coarse_correction!, shifted_amg_vcycle!, shifted_chiral_coarse_stencil
export gd_global_orthonormalize!, gd_coarse_odd_even_relative_residual
export gd_coarse_gmres, gd_adaptive_vcycle, gd_adaptive_amg_setup
export flexible_gmres!, harmonic_ritz_pencil, harmonic_ritz, generalized_davidson
export gd_eigenpair_audit

end # module LMP
