using Test
using Random
using LMP

# Reference Adam step for one real component, matching the convention of the
# CPU optimizer: moments are tracked separately for real and imaginary parts,
# the real part moves against the gradient and the imaginary part moves along
# it (the stored gradient is the conjugate Wirtinger derivative).
function reference_adam(w, g, m, v, t; α, β1, β2, ε)
    m = β1 * m + (1 - β1) * g
    v = β2 * v + (1 - β2) * g^2
    step = α * (m / (1 - β1^t)) / (sqrt(v / (1 - β2^t)) + ε)
    return step, m, v
end

@testset "Adam" begin
    @testset "advance_step! increments the global timestep" begin
        opt = Adam()
        @test opt.t == 0
        @test advance_step!(opt) == 1
        @test advance_step!(opt) == 2
        @test opt.t == 2
    end

    @testset "update! applies bias-corrected Adam and leaves t unchanged" begin
        α, β1, β2, ε = 0.01, 0.9, 0.999, 1.0e-8
        opt = Adam(α, β1, β2, ε)
        rng = Xoshiro(1)
        layers = [DenseLinearLayer(1, 2, false) for _ in 1:2]
        grads = [[LMP.Spin_matrix(randn(rng, ComplexF64, 4, 4)) for _ in 1:2] for _ in 1:2]

        W0 = [[copy(w.sm) for w in layer.W] for layer in layers]
        m = [[zeros(ComplexF64, 4, 4) for _ in 1:2] for _ in 1:2]
        v = [[zeros(ComplexF64, 4, 4) for _ in 1:2] for _ in 1:2]
        expected = deepcopy(W0)

        for t in 1:3
            @test advance_step!(opt) == t
            # Two layers in the same sweep must see the same timestep.
            for (l, layer) in enumerate(layers)
                update!(layer, opt, grads[l])
            end
            @test opt.t == t

            for l in 1:2, k in 1:2, idx in eachindex(expected[l][k])
                g = grads[l][k].sm[idx]
                sr, mr, vr = reference_adam(real(expected[l][k][idx]), real(g),
                                            real(m[l][k][idx]), real(v[l][k][idx]), t;
                                            α, β1, β2, ε)
                si, mi, vi = reference_adam(imag(expected[l][k][idx]), imag(g),
                                            imag(m[l][k][idx]), imag(v[l][k][idx]), t;
                                            α, β1, β2, ε)
                m[l][k][idx] = complex(mr, mi)
                v[l][k][idx] = complex(vr, vi)
                expected[l][k][idx] -= complex(sr, -si)
            end
            for l in 1:2, k in 1:2
                @test layers[l].W[k].sm ≈ expected[l][k] rtol = 1.0e-13
            end
        end
    end
end
