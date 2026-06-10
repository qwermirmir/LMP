module LMP




__precompile__(false)

include("elementry_base.jl")
include("gpu.jl")  # ✅ GPU 모듈 포함
include("Layer_types.jl")
include("activation.jl")
include("Optimizer.jl")  # ✅ Optimizer 모듈 포함
include("Layer_sequence.jl")
# include("BatchedBufferSequence.jl")
include("Gauge.jl")
include("DDalphaAMG.jl")


using .elementry_base 
using .gpu
using .Layer_types
using .activation
using .Optimizer
using .Layer_sequence
# using .BatchedBufferSequence
using .Gauge
using .DDalphaAMG
using DisplayAs
using JLD2
# ✅ `Optimizer`를 정확히 가져오고, 필요한 함수 import


# ✅ `LatticeDiracOperators` 방식 적용: import 후 export 실행
export Opt, Adam, update!, sync_layers!, save_checkpoint, load_checkpoint!
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
export compare_ddalphaamg_cpu_gpu
export gamma5!, chiral_project!, gamma5_sign
export CloverDiracOperator, compute_Fmunu, refresh_clover!
export train_timed
export BatchedFieldBuffer, buffer_from_fields, materialize_fields, materialize_fields!
export dense_forward_buffer!, pt_forward_buffer!, norm_forward_buffer!, forward_buffer_segment
export build_lmp_32pt_layers, load_lmp_model

function _default_lmp_paths(max_len::Int=4)
    P = Vector{Paths}(undef, 9)
    for i in 1:9
        if i <= 5
            P[i] = Paths([[i - 1]])
        else
            P[i] = Paths([[-(i - 5)]])
        end
    end

    dirs = [1, 2, 3, 4, -1, -2, -3, -4]
    lengths = [2^k for k in 1:floor(Int, log2(max_len))]
    if !(max_len in lengths)
        push!(lengths, max_len)
    end
    for d in dirs, len in lengths
        push!(P, Paths([fill(d, len)]))
    end
    return P
end

"""
    build_lmp_32pt_layers(U, x, D; max_len=4, init_gpu=true)

Builds the 16x32 32-PT Ritz model architecture used by `train_lmp.jl`.
This creates the layer objects only; checkpoint weights can then be loaded into
them with `load_lmp_model(path; U, x, D)`.
"""
function build_lmp_32pt_layers(U, x, D; max_len::Int=4, init_gpu::Bool=true)
    P = _default_lmp_paths(max_len)
    pl = length(P)

    dl = DenseLinearLayer(1, pl, true)
    PT_layers = [PTLayer(P, U, pl) for _ in 1:32]
    PT32 = PTLayer(P, U, 4)

    fe = fermionfield_Layer(x, 1)
    bs5 = BlockScalar_Layer(x, 4, 2)
    Norm = NormLayer()
    Norm2 = NormLayer()
    Ritz = Ritz_Layer(D)

    layers = [
        fe, dl,
        PT_layers[2:17]...,
        Norm,
        PT_layers[18:32]...,
        PT32, bs5, Norm2, Ritz,
    ]

    if init_gpu
        substitute_Layer!(layers)
        substitute_Layer!(layers)
    end

    return layers
end

function _infer_lmp_max_len_from_path(path::AbstractString, fallback::Int=4)
    m = match(r"maxlen(\d+)", basename(path))
    m === nothing && return fallback
    return parse(Int, m.captures[1])
end

"""
    load_lmp_model(path; layers=nothing, opt=Adam(), gpu=false, U=nothing, x=nothing, D=nothing, max_len=nothing)

Loads either legacy checkpoints that contain a `"layers"` key or newer W-only
checkpoints created by `save_checkpoint`.

Legacy files return their stored layers directly. W-only checkpoints need a live
`layers` object with the same architecture, or `U`, `x`, and `D` so the standard
32-PT Ritz architecture can be created automatically. If `max_len` is not given,
W-only checkpoints infer it from filenames like `maxlen5`, falling back to 4.
"""
function load_lmp_model(path::AbstractString;
                        layers=nothing,
                        opt::Adam=Adam(),
                        gpu::Bool=false,
                        U=nothing,
                        x=nothing,
                        D=nothing,
                        max_len=nothing)
    keys_in_file = JLD2.jldopen(path, "r") do f
        Set(String.(keys(f)))
    end

    if "layers" in keys_in_file
        loaded_layers = JLD2.load(path, "layers")
        if gpu
            substitute_Layer!(loaded_layers)
        end
        return (; layers=loaded_layers, opt, format=:legacy)
    end

    if "opt_t" in keys_in_file
        if layers === nothing
            if U !== nothing && x !== nothing && D !== nothing
                inferred_max_len = max_len === nothing ? _infer_lmp_max_len_from_path(path) : max_len
                layers = build_lmp_32pt_layers(U, x, D; max_len=inferred_max_len)
            else
                ks = sort!(collect(keys_in_file))
                preview = join(ks[1:min(end, 8)], ", ")
                error("""
                $path is a W-only checkpoint, so it does not contain a full `"layers"` object.
                Pass either existing `layers`, or pass `U`, `x`, and `D` so the standard
                32-PT model can be created automatically:

                    model = load_lmp_model(path; U=U, x=x, D=D, opt=opt)

                Available keys start with: $preview
                """)
            end
        end
        load_checkpoint!(path, layers, opt)
        return (; layers, opt, format=:checkpoint)
    end

    error("Unsupported LMP checkpoint format in $path. Available keys: $(collect(keys_in_file))")
end
end  # module LMP
