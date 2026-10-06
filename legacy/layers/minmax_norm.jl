# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/activation.jl.

mutable struct minmaxNormLayer <: Layer
    activation::ActivationFunction

    function minmaxNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end
end

function (layer::minmaxNormLayer)(x::WF3x1)
    
    
    min,max=minmax(x)

    x=x-min
    x=x/(max-min)

    return x
end

function gradient(layer::minmaxNormLayer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end

function DeltaRule(layer::minmaxNormLayer, ::Identity,z::WF3x1, d::WF3x1)

    NX, NY, NZ, NT = z.NX, z.NY, z.NZ, z.NT
    NC, NG = z.NC, z.NG

    IX,IY,IZ,IT,IC,IG = z.NX, z.NY, z.NZ, z.NT, z.NC, z.NG
    AX, AY, AZ, AT, AC, AG = z.NX, z.NY, z.NZ, z.NT, z.NC, z.NG

    first_val = z[1, 1, 1, 1, 1, 1]
    min = abs(first_val)
    max = abs(first_val)

    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            val = abs(z[ic, ix, iy, iz, it, ig])
                            if abs(val) < abs(min)
                                min = abs(val)
                                IX,IY,IZ,IT,IC,IG = ix, iy, iz, it, ic, ig
                            end
                            if abs(val) > abs(max)
                                max = abs(val)
                                AX, AY, AZ, AT, AC, AG = ix, iy, iz, it, ic, ig
                            end
                        end
                    end
                end
            end
        end
    end

    r=max-min

    result=similar(d)

    result=d/r


    zmin = z[IX,IY,IZ,IT,IC,IG]
    zmax = z[AX,AY,AZ,AT,AC,AG]

    result[IX,IY,IZ,IT,IC,IG] += (conj(zmin)/abs(zmin)) * inner_product(conj(d), z - max) / (r^2)

    result[AX,AY,AZ,AT,AC,AG] += (conj(zmax)/abs(zmax)) * inner_product(conj(d), min - z) / (r^2)


    return result
end
