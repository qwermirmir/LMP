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

function covariance_setup()
    U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="hot", cuda=false,
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

@testset "gauge covariance" begin
    (; U, V, g, x, gx, rng) = covariance_setup()

    @testset "Hopping_operator, path $steps" for steps in
            ([1], [2], [3], [4], [-1], [-2], [-3], [-4], [1, 2], [3, -4, -4])
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
