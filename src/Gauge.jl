


module  Gauge



# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra





export gUgshift!

# function gaugetransformation(A::Array{G,1}, U::Array{G,1}) where {G<:AbstractGaugefields}

#     new_U=Array{G,1}(undef,4)

#     for i in 1:4
#         Ushift=shift_U(Ucopy[i], i)
#     end
#     return U*A*U'

# end

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


function gUgshift!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
    ) where {NC,Dim,T<:AbstractGaugefields}
    
    for μ in 1:Dim
        g_shift = shift_U(g, μ) # g_shift(n) = g(n+μ)
        mul!(temp, g, U[μ]) # temp = g(n) * U[μ](n)
        mul!(U[μ], temp, g_shift') # U_transform[μ] = g(n) * U[μ](n) * g(n+μ)†
    end
    set_wing_U!(U)
end


# function gUgshift!(
#     Uout::Array{T,1},
#     Uin::Array{T,1},
#     g::AbstractGaugefields{NC,Dim},
#     temp::AbstractGaugefields{NC,Dim},
#     ) where {NC,Dim,T<:AbstractGaugefields}
    
#     for μ in 1:Dim
#         g_shift = shift_U(g, μ) # g_shift(n) = g(n+μ)
#         mul!(temp, g, Uin[μ]) # temp = g(n) * U[μ](n)
#         mul!(Uout[μ], temp, g_shift') # U_transform[μ] = g(n) * U[μ](n) * g(n+μ)†
#     end
#     set_wing_U!(Uout)
# end














end # module Gauge
