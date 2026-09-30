# Finite-difference checks of the backward pass on the golden models.
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

# On the asymmetric lattice single-entry gradients are about 1e-5, so rounding
# in the loss dominates a small step: at h = 1e-5 the central difference is off
# by about 1e-5 relative, and the error falls like 1/h up to h = 1e-3, where it
# is below 2e-7.
const FD_STEP = 1.0e-3
const FD_PROBES_PER_LAYER = 3

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

# All trainable arrays of a layer, and the matching gradient arrays (copied,
# because some layers return a cached buffer).
weight_arrays(layer::fermionfield_Layer, g) = ([layer.x[1].f], [copy(g[1].f)])
weight_arrays(layer::DenseLinearLayer, g) = ([W.sm for W in layer.W], [copy(G.sm) for G in g])
weight_arrays(layer::PTLayer, g) = ([W.sm for W in layer.W], [copy(G.sm) for G in only(g)])
weight_arrays(layer::Union{BlockScalar_Layer,BlockMatrix_Layer}, g) = ([layer.W], [copy(g)])

# Moves every weight of the layer at once along a random unit direction v and
# returns the central-difference slope together with the slope the gradient
# predicts, sum over weights of dL/dRe(w) Re(v) + dL/dIm(w) Im(v). A wrong
# gradient in any subset of entries changes the prediction.
function directional_slopes(layer, g, loss, rng)
    W, G = weight_arrays(layer, g)
    W0 = copy.(W)
    V = [randn(rng, ComplexF64, size(A)) for A in W]
    scale = 1 / sqrt(sum(sum(abs2, v) for v in V))
    foreach(v -> (v .*= scale), V)

    move!(t) = foreach((A, A0, v) -> (A .= A0 .+ t .* v), W, W0, V)
    move!(FD_STEP); lp = loss()
    move!(-FD_STEP); lm = loss()
    foreach(copyto!, W, W0)

    fd = (lp - lm) / (2FD_STEP)
    # expected_gradient is an involution, so it also maps g back to dL/dRe + i dL/dIm.
    predicted = sum(real(sum(conj.(expected_gradient.(Ref(layer), Gk)) .* vk)) for (Gk, vk) in zip(G, V))
    return fd, predicted
end

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

const MODELS = (
    ("BlockScalar model", golden_model),
    ("BlockMatrix model", blockmatrix_model),
    ("asymmetric BlockScalar model", asymmetric_model),
    ("asymmetric BlockMatrix model", () -> blockmatrix_model(dims=ASYM_DIMS, max_len=ASYM_MAX_LEN)),
)

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
                fd, predicted = directional_slopes(layers[i], grads[i], loss, rng)
                @test isapprox(predicted, fd; rtol=1.0e-5)
            end
        end
    end

    @testset "one Adam step lowers the loss, $name" for (name, build) in MODELS
        (; layers, D) = build()
        loss_before, loss_after = adam_step_losses(layers, D)
        @test loss_after < loss_before
    end
end
