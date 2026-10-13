# Text written to the training log: run metadata, layer summaries and loss
# metrics.

# import LatticeDiracOperators: Dirac_Operators


# const Dirac_operator =LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}
# const DagDirac_operator =LatticeDiracOperators.Dirac_operators.Adjoint_Wilson_operator{LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}



_metadata_value_string(x) = x === nothing ? "nothing" : string(x)

function _print_metadata_block(io::IO, title::AbstractString, rows)
    println(io, "# --- $(title) ---")
    for row in rows
        if row isa Pair
            println(io, "# ", row.first, ",", _metadata_value_string(row.second))
        else
            println(io, "# ", _metadata_value_string(row))
        end
    end
end

function _layer_shape_summary(layer)
    if layer isa DenseLinearLayer
        return "In=$(layer.In), Out=$(layer.Out), cuda=$(layer.cuda), W=$(length(layer.W))"
    elseif layer isa PTLayer
        return "paths=$(length(layer.paths)), Out=$(layer.Out), W=$(length(layer.W))"
    elseif layer isa fermionfield_Layer
        return "In=$(layer.In)"
    elseif layer isa NormLayer
        return ""
    elseif layer isa Ritz_Layer
        return "Out=$(layer.Out)"
    elseif layer isa ZeroMode_Layer
        return "Out=$(layer.Out)"
    elseif layer isa BlockScalar_Layer
        return "W=$(size(layer.W)), r=$(layer.r), residual=$(layer.residual)"
    elseif layer isa LossSumLayer
        return "w1=$(layer.w1), w2=$(layer.w2), l1=$(nameof(typeof(layer.l1))), l2=$(nameof(typeof(layer.l2)))"
    else
        parts = String[]
        hasproperty(layer, :W) && push!(parts, "W=$(size(getproperty(layer, :W)))")
        hasproperty(layer, :Out) && push!(parts, "Out=$(getproperty(layer, :Out))")
        hasproperty(layer, :r) && push!(parts, "r=$(getproperty(layer, :r))")
        return join(parts, ", ")
    end
end

function _coarse_loss_summary(layer)
    !hasproperty(layer, :coarse_tol) && return nothing
    parts = [
        "operator=$(getproperty(layer, :operator))",
        "objective=$(getproperty(layer, :objective))",
        "gain_eps=$(getproperty(layer, :gain_eps))",
        "objective_scale=$(getproperty(layer, :objective_scale))",
        "block_size=$(getproperty(layer, :block_size))",
        "use_chirality=$(getproperty(layer, :use_chirality))",
        "tol=$(getproperty(layer, :coarse_tol))",
        "maxiter=$(getproperty(layer, :coarse_maxiter))",
        "restart=$(getproperty(layer, :coarse_restart))",
        "refresh_error=$(getproperty(layer, :refresh_error))",
        "smooth_steps=$(getproperty(layer, :error_smooth_steps))",
        "cycle_smoother=$(getproperty(layer, :cycle_smoother))",
        "cycle_omega=$(getproperty(layer, :cycle_omega))",
        "cycle_pre_smooth_steps=$(getproperty(layer, :cycle_pre_smooth_steps))",
        "cycle_post_smooth_steps=$(getproperty(layer, :cycle_post_smooth_steps))",
        "block_normalize=$(getproperty(layer, :block_normalize))",
        "tangent_project=$(getproperty(layer, :tangent_project))",
        "orth_pullback=$(getproperty(layer, :orthonormalization_pullback))",
    ]
    hasproperty(layer, :_replay_buffer) &&
        push!(parts, "replay_buffer_size=$(get(ENV, "COARSE_LOSS_REPLAY_BUFFER_SIZE", "0"))")
    hasproperty(layer, :_replay_buffer) &&
        push!(parts, "replay_errors_per_forward=$(get(ENV, "COARSE_LOSS_REPLAY_ERRORS_PER_FORWARD", "0"))")
    hasproperty(layer, :_replay_buffer) &&
        push!(parts, "replay_store_per_forward=$(get(ENV, "COARSE_LOSS_REPLAY_STORE_PER_FORWARD", get(ENV, "COARSE_LOSS_REPLAY_ERRORS_PER_FORWARD", "0")))")
    return join(parts, ", ")
end

function _collect_nested_loss_layers!(rows::Vector{String}, layer, prefix::String)
    if layer isa LossSumLayer
        push!(rows, "$(prefix)LossSumLayer: w1=$(layer.w1), w2=$(layer.w2)")
        _collect_nested_loss_layers!(rows, layer.l1, prefix * "  l1.")
        _collect_nested_loss_layers!(rows, layer.l2, prefix * "  l2.")
    else
        summary = _layer_shape_summary(layer)
        line = "$(prefix)$(nameof(typeof(layer)))"
        !isempty(summary) && (line *= ": $(summary)")
        coarse = _coarse_loss_summary(layer)
        coarse !== nothing && (line *= ": $(coarse)")
        push!(rows, line)
    end
    return rows
end

function _layer_metadata_rows(layers)
    rows = String[]
    for (i, layer) in enumerate(layers)
        summary = _layer_shape_summary(layer)
        line = "layer[$(i)],$(nameof(typeof(layer)))"
        !isempty(summary) && (line *= ",$(summary)")
        coarse = _coarse_loss_summary(layer)
        coarse !== nothing && (line *= ",$(coarse)")
        push!(rows, line)
        if layer isa LossSumLayer
            _collect_nested_loss_layers!(rows, layer.l1, "  layer[$(i)].l1.")
            _collect_nested_loss_layers!(rows, layer.l2, "  layer[$(i)].l2.")
        end
    end
    return rows
end

function _write_training_metadata(io::IO, layers, params, gauge_paths, test_paths, gauge_format, opt,
                                  epochs, epoch_offset, batch, N, logfile, checkpoint_every,
                                  activation_checkpoint_interval, run_metadata)
    println(io, "# --- Run Metadata ---")
    println(io, "# started_at,", Libc.strftime("%Y-%m-%d %H:%M:%S", time()))
    println(io, "# cwd,", pwd())
    println(io, "# julia,", VERSION)
    println(io, "# cuda_visible_devices,", get(ENV, "CUDA_VISIBLE_DEVICES", ""))
    println(io, "# gpu_functional,", CUDA.functional())
    println(io, "# logfile,", logfile)
    println(io, "# epochs_requested,", epochs)
    println(io, "# epoch_offset,", epoch_offset)
    println(io, "# batch,", batch)
    println(io, "# depth,", length(layers))
    println(io, "# configs_per_epoch,", N)
    println(io, "# full_epochs,", (epochs * batch) / N)
    println(io, "# gauge_format,", gauge_format)
    println(io, "# train_gauge_count,", length(gauge_paths))
    !isempty(gauge_paths) && println(io, "# train_gauge_first,", first(gauge_paths))
    !isempty(gauge_paths) && println(io, "# train_gauge_last,", last(gauge_paths))
    if test_paths !== nothing
        println(io, "# test_gauge_count,", length(test_paths))
        !isempty(test_paths) && println(io, "# test_gauge_first,", first(test_paths))
        !isempty(test_paths) && println(io, "# test_gauge_last,", last(test_paths))
    else
        println(io, "# test_gauge_count,0")
    end
    println(io, "# optimizer,", nameof(typeof(opt)))
    hasproperty(opt, :α) && println(io, "# learning_rate,", getproperty(opt, :α))
    println(io, "# checkpoint_every,", checkpoint_every)
    activation_checkpoint_interval > 0 &&
        println(io, "# activation_checkpoint_interval,", activation_checkpoint_interval)
    if params isa AbstractDict
        _print_metadata_block(io, "Dirac/Physics Parameters", sort(collect(params); by=x -> string(x.first)))
    else
        println(io, "# params,", params)
    end
    if run_metadata !== nothing
        if run_metadata isa AbstractDict
            _print_metadata_block(io, "Experiment Metadata", sort(collect(run_metadata); by=x -> string(x.first)))
        else
            _print_metadata_block(io, "Experiment Metadata", run_metadata)
        end
    end
    _print_metadata_block(io, "Layer Stack", _layer_metadata_rows(layers))
    println(io, "# --- End Metadata ---")
end

function _coarse_metric_rows!(rows::Vector{String}, layer, prefix::String="")
    if layer isa LossSumLayer
        _coarse_metric_rows!(rows, layer.l1, prefix * "l1.")
        _coarse_metric_rows!(rows, layer.l2, prefix * "l2.")
    elseif hasproperty(layer, :ratio_value) && hasproperty(layer, :gain_value)
        pfx = isempty(prefix) ? "" : prefix
        push!(rows, "$(pfx)ratio=$(getproperty(layer, :ratio_value))")
        push!(rows, "$(pfx)gain=$(getproperty(layer, :gain_value))")
        hasproperty(layer, :orth_residual) &&
            push!(rows, "$(pfx)coarse_orth_rel=$(getproperty(layer, :orth_residual))")
        hasproperty(layer, :replay_used) &&
            push!(rows, "$(pfx)replay_used=$(getproperty(layer, :replay_used))")
        hasproperty(layer, :replay_stored) &&
            push!(rows, "$(pfx)replay_stored=$(getproperty(layer, :replay_stored))")
    end
    return rows
end

function _loss_metrics_text(layer, current_loss=nothing)
    rows = _coarse_metric_rows!(String[], layer)
    if layer isa LossSumLayer && current_loss !== nothing
        if hasproperty(layer.l1, :loss_value)
            push!(rows, "l1.loss=$(getproperty(layer.l1, :loss_value))")
        end
        if hasproperty(layer.l2, :loss_value)
            l2_loss = getproperty(layer.l2, :loss_value)
            push!(rows, "l2.loss=$(l2_loss)")
            if layer.w1 != 0
                total = real(current_loss)
                push!(rows, "l1.loss_inferred=$((total - layer.w2 * real(l2_loss)) / layer.w1)")
            end
        end
    end
    return isempty(rows) ? "" : ", " * join(rows, ", ")
end
