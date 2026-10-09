# Gauge transformations of link fields, V_mu(x) = g(x) U_mu(x) g(x+mu)^dagger.

function gUgshift(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
    ) where {NC,Dim,T<:AbstractGaugefields}

    U_transform = similar(U)
    
    for μ in 1:Dim
        g_shift = shift_U(g, μ) # g_shift(n) = g(n+μ)
        mul!(temp, g, U[μ]) # temp = g(n) * U[μ](n)
        mul!(U_transform[μ], temp, g_shift') # U_transform[μ] = g(n) * U[μ](n) * g(n+μ)†
    end
    
    set_wing_U!(U_transform)
    return U_transform

end

# The in-place gUgshift!(U, g, temp) comes from Gaugefields, which exports the
# same function since 1.1. LMP re-exports it; a second definition here would
# make the name ambiguous for code that uses both packages.
