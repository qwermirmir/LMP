# Constructing the standard LMP model and loading a trained one.

"""
    build_lmp_32pt_layers(U, x, D; max_len=4, init_gpu=true)

Builds the 16x32 32-PT Ritz model architecture used by `train_lmp.jl`.
This creates the layer objects only; checkpoint weights can then be loaded into
them with `load_lmp_model(path; U, x, D)`.
"""
function build_lmp_32pt_layers(U, x, D; max_len::Int=4, num_eig::Int=4, block_r::Int=4, init_gpu::Bool=true)
    P = _default_lmp_paths(max_len)
    pl = length(P)

    dl = DenseLinearLayer(1, pl, true)
    PT_layers = [PTLayer(P, U, pl) for _ in 1:32]
    PT32 = PTLayer(P, U, num_eig)

    fe = fermionfield_Layer(x, 1)
    bs5 = BlockScalar_Layer(x, num_eig, block_r)
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

function _infer_lmp_num_eig_from_path(path::AbstractString, fallback::Int=4)
    m = match(r"\((\d+)eig\)", basename(path))
    m === nothing && return fallback
    return parse(Int, m.captures[1])
end

"""
    load_lmp_model(path; layers=nothing, opt=Adam(), gpu=false, U=nothing, x=nothing, D=nothing, max_len=nothing, num_eig=nothing)

Loads either legacy checkpoints that contain a `"layers"` key or newer W-only
checkpoints created by `save_checkpoint`.

Legacy files return their stored layers directly. W-only checkpoints need a live
`layers` object with the same architecture, or `U`, `x`, and `D` so the standard
32-PT Ritz architecture can be created automatically. If `max_len` is not given,
W-only checkpoints infer it from filenames like `maxlen5`, falling back to 4.
If `num_eig` is not given, W-only checkpoints infer it from filenames like
`(6eig)`, falling back to 4.
"""
function load_lmp_model(path::AbstractString;
                        layers=nothing,
                        opt::Adam=Adam(),
                        gpu::Bool=false,
                        U=nothing,
                        x=nothing,
                        D=nothing,
                        max_len=nothing,
                        num_eig=nothing)
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
                inferred_num_eig = num_eig === nothing ? _infer_lmp_num_eig_from_path(path) : num_eig
                layers = build_lmp_32pt_layers(U, x, D; max_len=inferred_max_len, num_eig=inferred_num_eig)
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
