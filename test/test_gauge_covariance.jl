# Gauge covariance of the parallel-transport building blocks.
#
# Under V_mu(x) = g(x) U_mu(x) g(x + mu)^dagger, a covariant operator O satisfies
#     O_V(g psi) = g O_U(psi).

using Test
using Random
using LinearAlgebra
using Gaugefields
using LatticeDiracOperators
using LMP

_relerr(a, b) = norm(a.f .- b.f) / norm(a.f)

# The second lattice has a different extent in every direction, so a swapped
# direction or index changes the result.
const LATTICES = ((4, 4, 4, 4), (4, 6, 8, 10))

const TEST_PATHS = ([1], [2], [3], [4], [-1], [-2], [-3], [-4], [1, 2], [3, -4, -4],
                    [4, 4, 4, 4, 4, 4])

# One parallel-transport step written directly from the definition,
#     +mu: out(n) = U_mu(n - mu)^dagger psi(n - mu),
#     -mu: out(n) = U_mu(n) psi(n + mu),
# with antiperiodic boundary conditions in time.
function reference_step(U, f, step)
    mu = abs(step)
    offset = step > 0 ? -1 : 1
    Umu = U[mu].U
    dims = size(f)[2:5]
    out = zero(f)
    for n in CartesianIndices(dims)
        m = collect(Tuple(n))
        m[mu] += offset
        sign = 1
        if !(1 <= m[mu] <= dims[mu])
            m[mu] = mod1(m[mu], dims[mu])
            mu == 4 && (sign = -1)
        end
        link = step > 0 ? adjoint(Umu[:, :, m...]) : Umu[:, :, Tuple(n)...]
        for ig in axes(f, 6)
            out[:, Tuple(n)..., ig] = sign * link * f[:, m..., ig]
        end
    end
    return out
end

reference_path(U, f, steps) = foldl((g, step) -> reference_step(U, g, step), steps; init=f)

function covariance_setup(dims)
    U = Initialize_Gaugefields(3, 0, dims...; condition="hot", cuda=false,
                               randomnumber="Reproducible")
    # Any SU(3)-valued field is a valid gauge transformation; reuse a link field.
    g = deepcopy(U[4])
    V = deepcopy(U)
    gUgshift!(V, g, similar(g))

    rng = Xoshiro(7)
    x = Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    x.f .= randn(rng, ComplexF64, size(x.f))
    gx = similar(x)
    mul!(gx, g, x)
    return (; U, V, g, x, gx, rng)
end

@testset "gauge covariance, lattice $dims" for dims in LATTICES
    (; U, V, g, x, gx, rng) = covariance_setup(dims)

    @testset "Hopping_operator matches the definition, path $steps" for steps in TEST_PATHS
        out = Hopping_operator(U, Paths([steps]))(x)[1]
        expected = reference_path(U, x.f, steps)
        @test norm(out.f .- expected) / norm(expected) < 1.0e-13
    end

    @testset "Hopping_operator, path $steps" for steps in TEST_PATHS
        P = Paths([steps])
        lhs = similar(x)
        mul!(lhs, g, Hopping_operator(U, P)(x)[1])
        rhs = Hopping_operator(V, P)(gx)[1]
        @test _relerr(lhs, rhs) < 1.0e-12
    end

    @testset "reverse_paths inverts each path" begin
        for P in LMP._default_lmp_paths(4)
            forward = deepcopy(Hopping_operator(U, P)(x)[1])
            back = Hopping_operator(U, reverse_paths(P))(forward)[1]
            @test _relerr(x, back) < 1.0e-12
        end
    end

    @testset "PTLayer" begin
        P = LMP._default_lmp_paths(4)
        nout = 3
        layer_U = PTLayer(P, U, nout)
        layer_V = PTLayer(P, V, nout)
        for k in eachindex(layer_U.W)
            layer_V.W[k].sm .= layer_U.W[k].sm
        end

        inputs = [similar(x) for _ in P]
        foreach(f -> (f.f .= randn(rng, ComplexF64, size(f.f))), inputs)
        g_inputs = [similar(x) for _ in P]
        foreach(i -> mul!(g_inputs[i], g, inputs[i]), eachindex(inputs))

        out_U = deepcopy(layer_U(inputs))
        out_V = layer_V(g_inputs)
        for j in 1:nout
            lhs = similar(x)
            mul!(lhs, g, out_U[j])
            @test _relerr(lhs, out_V[j]) < 1.0e-12
        end
    end
end
