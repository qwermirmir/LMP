module SubspaceSimilarity

using LinearAlgebra

export orthonormal_basis, subspace_similarity, per_mode_capture

function orthonormal_basis(X::AbstractMatrix; rank_tol::Real=sqrt(eps(Float64)))
    size(X, 2) > 0 || throw(ArgumentError("input must contain at least one vector"))
    F = svd(Matrix{ComplexF64}(X); full=false)
    threshold = Float64(rank_tol) * max(first(F.S), 1.0)
    numerical_rank = count(>(threshold), F.S)
    expected_rank = size(X, 2)
    numerical_rank == expected_rank || throw(ArgumentError(
        "input vectors have numerical rank $numerical_rank, expected $expected_rank"))
    return F.U[:, 1:numerical_rank]
end

function subspace_similarity(QE::AbstractMatrix, QL::AbstractMatrix)
    E = orthonormal_basis(QE)
    L = orthonormal_basis(QL)
    size(E, 1) == size(L, 1) || throw(DimensionMismatch(
        "bases have different ambient dimensions: $(size(E, 1)) and $(size(L, 1))"))
    C = E' * L
    singular_values = clamp.(svdvals(C), 0.0, 1.0)
    return (;
        cross_gram=C,
        singular_values,
        principal_angles_deg=rad2deg.(acos.(singular_values)),
        mean_capture=sum(abs2, C) / size(E, 2),
        worst_capture=minimum(singular_values)^2,
    )
end

function per_mode_capture(QE::AbstractMatrix, QL::AbstractMatrix)
    E = orthonormal_basis(QE)
    L = orthonormal_basis(QL)
    size(E, 1) == size(L, 1) || throw(DimensionMismatch(
        "bases have different ambient dimensions: $(size(E, 1)) and $(size(L, 1))"))
    return vec(sum(abs2, L' * E; dims=1))
end

end
