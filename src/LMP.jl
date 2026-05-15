module LMP




__precompile__(false)

include("elementry_base.jl")
include("gpu.jl")  # ✅ GPU 모듈 포함
include("Layer_types.jl")
include("activation.jl")
include("Optimizer.jl")  # ✅ Optimizer 모듈 포함
include("Layer_sequence.jl")
include("Gauge.jl")
include("DDalphaAMG.jl")


using .elementry_base 
using .gpu
using .Layer_types
using .activation
using .Optimizer
using .Layer_sequence
using .Gauge
using .DDalphaAMG
using DisplayAs
# ✅ `Optimizer`를 정확히 가져오고, 필요한 함수 import


# ✅ `LatticeDiracOperators` 방식 적용: import 후 export 실행
export Opt, Adam, update!, sync_layers!
export Hopping_operator
export Onestep
export Paths
export PTC
export CPTC
export Random_SpinM
export Generate_weight
export Spin_matrix
export Spin_matrix_GPU
export norm2
export reverse_paths
export Lossfunction
export PTC_Layer
export CPTC_Layer
export forward_propagation
export train 
export Identity
export backward_propagation
export Clifford
export inner_product
export Attention_Layer
export HalfPAttention_Layer
export Matmul!
export preD
export Operator
export doubletrain
export Softmax
export NormLayer
export test_Layer
export FullPTCAttention_Layer
export mean 
export Spin_matrix_GPU
export Random_SpinM_GPU
export PrallelTransportAttention_Layer
#----------------------------------------------
export gUgshift!
export xavier_init!
export complexNormLayer
export minmaxNormLayer
export minmax
export gradient 
export standardization
export DeltaRule
export Dirac_Layer
export ModRelu
export g2vector_generator!
export siteweight_Layer
export RayleighQuotient_Layer
export L2Norm_Layer
export PtcRayleighQuotient_Layer
export HoppingRayleighQuotient_Layer
export DenseLinearLayer
export PTLayer
export addmul!
export fermionfield_Layer
export Orthogonal_Layer
export LossSumLayer
export Ritz_Layer
export Convertsm
export substitute_fermion!
export gpuaddmul!
export GramSchmidt_Layer
export BlockScalar_Layer
export conj!
export gpuconj!
export backward_propagation!,forward_propagation!
export zero_wf!
export refresh_fermionfield!, refresh_model!
export substitute_Layer!
export blockize
export unblockize
export DDalphaAMG_Setup, CoarseVector
export setup_DDalphaAMG, solve_DDalphaAMG, vcycle
export restrict_f2c, prolongate_c2f, coarse_apply
export restrict_f2c!, prolongate_c2f!, coarse_apply!
export precompute_coarse_operator!
export mr_smooth!, sap_smooth!
export block_orthonormalize!,deflated_initial_guess,_to_cpu,_to_gpu,_to_cpu_gauge
export gamma5!, chiral_project!, gamma5_sign
export CloverDiracOperator, compute_Fmunu, refresh_clover!
export train_timed
end  # module LMP

