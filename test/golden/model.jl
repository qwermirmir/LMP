# Small deterministic LMP model used by the golden regression test.
#
# The model mirrors the layer sequence of `build_lmp_32pt_layers` at a size that
# runs on CPU in seconds: 4^4 lattice, short paths, three PT layers.
#
# All trainable weights and the input fermion field are overwritten from a
# dedicated `Xoshiro` stream after construction, so the reference values do not
# depend on how many random numbers the layer constructors consume.

using Random
using LinearAlgebra
using Gaugefields
using LatticeDiracOperators
using LMP

const GOLDEN_SEED = 20260930
const GOLDEN_KAPPA = 0.141
const GOLDEN_NUM_EIG = 2
const GOLDEN_BLOCK_R = 2
const GOLDEN_DIMS = (4, 4, 4, 4)
const GOLDEN_MAX_LEN = 2

# Every lattice extent differs, so a swapped direction or index changes the
# result. Longer paths exercise multi-step hopping.
const ASYM_DIMS = (4, 6, 8, 10)
const ASYM_MAX_LEN = 4

function golden_gauge_and_dirac(dims=GOLDEN_DIMS)
    # "Reproducible" makes Gaugefields use a fixed StableRNG. The default
    # "Random" draws a fresh MersenneTwister and ignores Random.seed!.
    U = Initialize_Gaugefields(3, 0, dims...; condition="hot", cuda=false,
                               randomnumber="Reproducible")
    x = Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    params = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => GOLDEN_KAPPA,
        "faster version" => true,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, params)
    return U, x, D
end

function _fill_spin_weights!(W, rng, scale)
    for M in W
        for j in axes(M, 2), i in axes(M, 1)
            M[i, j] = scale * randn(rng, ComplexF64)
        end
    end
    return W
end

function golden_model(; dims=GOLDEN_DIMS, max_len=GOLDEN_MAX_LEN)
    U, x, D = golden_gauge_and_dirac(dims)
    P = LMP._default_lmp_paths(max_len)
    pl = length(P)

    fe = fermionfield_Layer(x, 1)
    dl = DenseLinearLayer(1, pl, false)
    pt1 = PTLayer(P, U, pl)
    pt2 = PTLayer(P, U, pl)
    pt_eig = PTLayer(P, U, GOLDEN_NUM_EIG)
    bs = BlockScalar_Layer(x, GOLDEN_NUM_EIG, GOLDEN_BLOCK_R)
    layers = [fe, dl, pt1, NormLayer(), pt2, pt_eig, bs, NormLayer(), Ritz_Layer(D)]

    rng = Xoshiro(GOLDEN_SEED + 1)
    fe.x[1].f .= randn(rng, ComplexF64, size(fe.x[1].f))
    _fill_spin_weights!(dl.W, rng, 1.0)
    for pt in (pt1, pt2, pt_eig)
        _fill_spin_weights!(pt.W, rng, 1 / sqrt(length(pt.paths)))
    end
    bs.W .= 1 .+ 0.1 .* randn(rng, ComplexF64, size(bs.W))

    return (; layers, U, x, D)
end

asymmetric_model() = golden_model(dims=ASYM_DIMS, max_len=ASYM_MAX_LEN)

_field_norm2(fields) = sum(real(dot(f.f, f.f)) for f in fields)

# Gradients come back as numbers, spin matrices, fermion fields, or nested
# vectors of these depending on the layer type.
_flatten_gradient(g::Number) = ComplexF64[g]
_flatten_gradient(g::AbstractArray{<:Number}) = ComplexF64.(vec(Array(g)))
_flatten_gradient(g::LMP.WF3x1) = ComplexF64.(vec(Array(g.f)))
_flatten_gradient(g) = reduce(vcat, (_flatten_gradient(m) for m in g))

"""
    golden_measure(layers = golden_model().layers)

Runs one forward pass, one backward pass, the weight gradients, and one Adam
update on `layers`, which end in a Ritz layer. Returns a `NamedTuple` of scalars
that summarize the result.
"""
function golden_measure(layers = golden_model().layers)
    D = layers[end].D
    depth = length(layers)
    gamma = Clifford()

    u, _ = forward_propagation(layers, D, D)
    delta = deepcopy(u[2:end])
    loss = forward_propagation!(layers, D, D, u)
    backward_propagation!(layers, u, 1, delta)

    activations = [_field_norm2(u[i]) for i in 2:depth]

    rng = Xoshiro(GOLDEN_SEED + 2)
    grad_norm2 = Float64[]
    grad_proj = ComplexF64[]
    grads = Vector{Any}(nothing, depth - 1)
    for i in 1:depth-1
        g = gradient(layers[i], layers[i].activation, u[i], delta[i], gamma)
        g === nothing && continue
        grads[i] = g
        flat = _flatten_gradient(g)
        push!(grad_norm2, sum(abs2, flat))
        for _ in 1:3
            push!(grad_proj, dot(randn(rng, ComplexF64, length(flat)), flat))
        end
    end

    opt = Adam(1.0e-3)
    advance_step!(opt)
    for i in 1:depth-1
        grads[i] === nothing && continue
        update!(layers[i], opt, grads[i])
    end
    loss_after_step = forward_propagation!(layers, D, D, u)

    return (;
        loss = ComplexF64(loss),
        activations,
        grad_norm2,
        grad_proj,
        loss_after_step = ComplexF64(loss_after_step),
    )
end

"""
    golden_train_step!(layers, opt)

One optimizer step as the training loop takes it: forward, backward, gradients,
`advance_step!` and `update!`. Returns the loss before the step and the squared
gradient norms.
"""
function golden_train_step!(layers, opt)
    D = layers[end].D
    depth = length(layers)
    gamma = Clifford()
    u, _ = forward_propagation(layers, D, D)
    delta = deepcopy(u[2:end])
    loss = real(forward_propagation!(layers, D, D, u))
    backward_propagation!(layers, u, 1, delta)
    grads = [gradient(layers[i], layers[i].activation, u[i], delta[i], gamma) for i in 1:depth-1]
    advance_step!(opt)
    for i in 1:depth-1
        grads[i] === nothing || update!(layers[i], opt, grads[i])
    end
    grad_norm2 = [sum(abs2, _flatten_gradient(g)) for g in grads if g !== nothing]
    return (; loss, grad_norm2)
end
