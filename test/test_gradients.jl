# Finite-difference checks of the backward pass on the golden model.
#
# `gradient` returns, for every complex weight w,
#     g = dL/dRe(w) - i dL/dIm(w),
# so steepest descent is w -= eta * conj(g). The Adam updates in Optimizer.jl
# and gpu.jl rely on this convention.
#
# BlockMatrix_Layer is the exception: it returns the conjugate,
# g = dL/dRe(w) + i dL/dIm(w), and its Adam update (CPU and
# `_cudakernel_adam_complex_standard!`) is written for that.

using Test
using Random

include(joinpath(@__DIR__, "golden", "model.jl"))

const FD_STEP = 1.0e-5
const FD_PROBES_PER_LAYER = 3

# Same model with the BlockScalar layer replaced by a BlockMatrix layer.
function blockmatrix_model()
    model = golden_model()
    bm = BlockMatrix_Layer(model.x, GOLDEN_NUM_EIG, GOLDEN_BLOCK_R)
    rng = Xoshiro(GOLDEN_SEED + 4)
    bm.W .+= 0.1 .* randn(rng, ComplexF64, size(bm.W))
    i = findfirst(l -> l isa BlockScalar_Layer, model.layers)
    model.layers[i] = bm
    return model
end

function analytic_gradients(layers, D)
    u, _ = forward_propagation(layers, D, D)
    delta = deepcopy(u[2:end])
    forward_propagation!(layers, D, D, u)
    backward_propagation!(layers, u, 1, delta)
    gamma = Clifford()
    grads = [gradient(layers[i], layers[i].activation, u[i], delta[i], gamma)
             for i in 1:length(layers)-1]
    return grads, u
end

# Central-difference estimate of dL/dRe(w) + i dL/dIm(w).
function fd_derivative(loss, get, set!)
    w0 = get()
    set!(w0 + FD_STEP); lp = loss()
    set!(w0 - FD_STEP); lm = loss()
    d_re = (lp - lm) / (2FD_STEP)
    set!(w0 + im * FD_STEP); lp = loss()
    set!(w0 - im * FD_STEP); lm = loss()
    d_im = (lp - lm) / (2FD_STEP)
    set!(w0)
    return complex(d_re, d_im)
end

# A probe is (get, set!, analytic) for one randomly chosen scalar weight. The
# analytic value is copied out because some layers return a cached buffer.
function _array_probes(A, g, rng, n)
    return map(1:n) do _
        i = rand(rng, CartesianIndices(A))
        (() -> A[i], v -> (A[i] = v), g[i])
    end
end

function _spin_probes(W, g, rng, n)
    return map(1:n) do _
        k = rand(rng, eachindex(W))
        i = rand(rng, CartesianIndices(W[k].sm))
        (() -> W[k].sm[i], v -> (W[k].sm[i] = v), g[k].sm[i])
    end
end

probes(layer::fermionfield_Layer, g, rng, n) = _array_probes(layer.x[1].f, g[1].f, rng, n)
probes(layer::DenseLinearLayer, g, rng, n) = _spin_probes(layer.W, g, rng, n)
probes(layer::PTLayer, g, rng, n) = _spin_probes(layer.W, only(g), rng, n)
probes(layer::Union{BlockScalar_Layer,BlockMatrix_Layer}, g, rng, n) =
    _array_probes(layer.W, g, rng, n)

# Maps dL/dRe + i dL/dIm to the convention the layer's gradient uses.
expected_gradient(layer, fd) = conj(fd)
expected_gradient(::BlockMatrix_Layer, fd) = fd

function adam_step_losses(layers, D)
    grads, u = analytic_gradients(layers, D)
    loss_before = real(forward_propagation!(layers, D, D, u))
    opt = Adam(1.0e-3)
    advance_step!(opt)
    for i in eachindex(grads)
        grads[i] === nothing || update!(layers[i], opt, grads[i])
    end
    loss_after = real(forward_propagation!(layers, D, D, u))
    return loss_before, loss_after
end

const MODELS = (("BlockScalar model", golden_model), ("BlockMatrix model", blockmatrix_model))

@testset "gradients" begin
    @testset "finite differences, $name" for (name, build) in MODELS
        (; layers, D) = build()
        grads, u = analytic_gradients(layers, D)
        loss() = real(forward_propagation!(layers, D, D, u))
        rng = Xoshiro(GOLDEN_SEED + 3)
        for i in eachindex(grads)
            grads[i] === nothing && continue
            @testset "layer $i $(nameof(typeof(layers[i])))" begin
                for (get, set!, g) in probes(layers[i], grads[i], rng, FD_PROBES_PER_LAYER)
                    fd = fd_derivative(loss, get, set!)
                    @test isapprox(g, expected_gradient(layers[i], fd); rtol=1.0e-5)
                end
            end
        end
    end

    @testset "one Adam step lowers the loss, $name" for (name, build) in MODELS
        (; layers, D) = build()
        loss_before, loss_after = adam_step_losses(layers, D)
        @test loss_after < loss_before
    end
end
