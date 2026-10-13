# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/elementry_base.jl.

# (μ,ν) pair → σ 배열의 인덱스
@inline function _sigma_index(mu::Int, nu::Int)
    # (1,2)→1, (1,3)→2, (1,4)→3, (2,3)→4, (2,4)→5, (3,4)→6
    mu < nu || error("mu must be < nu")
    return (mu - 1) * (7 - mu) ÷ 2 + (nu - mu)
end

function Onestep()
    dirs = (0, 1, 2, 3, 4, -1, -2, -3, -4)
    single_steps = [[d] for d in dirs]  # 각 값을 리스트로 감싸기
    return Paths(single_steps)  # 가변 인자(...)를 제거
end

function PTC(W::Vector{Spin_matrix},hop::Hopping_operator,x::WF3x1) 

    result=similar(x)



    fields=hop(x)

    if length(W)==1
        for i in 1:length(fields)
            result+=W[1]*fields[i]
        end
    end


        


    if length(W)==length(fields)

        for i in 1:length(fields)
            result+=W[i]*fields[i]
        end
    end

    
    return result
    

end    

function CPTC(hop::Hopping_operator,W::Vector{Spin_matrix},f::WF3x1)
    
    s=length(hop)
    cW=Clifford(W)
    result=similar(f)
    if length(cW)==s
        for i in 1:s
            f+=cW[i]*hop(f)[i]
        end
    end

    if length(cW)==1
        for i in 1:s
            f+=cW[1]*hop(f)[i]
        end
    end

    return result
end

function standardization(x::WF3x1)
    mu = mean(x)
    x = x - mu

    real = (x + conj(x)) / 2
    imag = (x - conj(x)) / 2im

    N = x.NX * x.NY * x.NZ * x.NT * x.NC * x.NG
    σr = sqrt(norm2(real) / N)
    σi = sqrt(norm2(imag) / N)

    x = real / σr + im * imag / σi

    return x
end

function g2vector_generator!(x::WF3x1; p::Real=0.5, rng::AbstractRNG=Random.default_rng())
    @assert 0 ≤ p ≤ 1 "p must be in [0,1]"
    NX, NY, NZ, NT, NC, NG = x.NX, x.NY, x.NZ, x.NT, x.NC, x.NG
    @inbounds for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            x[ic, ix, iy, iz, it, ig] = (rand(rng) < p) ? 1 : 0
                        end
                    end
                end
            end
        end
    end
    return x
end

function rand01_like(x::WF3x1; p::Real=0.5, rng::AbstractRNG=Random.default_rng())
    r = deepcopy(x)
    fill01!(r; p=p, rng=rng)
    return r
end

function spincolortrace(x::WF3x1,y::WF3x1)
    @assert size(x) == size(y) "Input vectors must have the same dimensions"
    NX, NY, NZ, NT, NC, NG = x.NX, x.NY, x.NZ, x.NT, x.NC, x.NG
    L=NX * NY * NZ * NT
    trace_sum = Array{ComplexF64}(undef, NX, NY, NZ, NT)
 
        @inbounds    for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        local_trace = 0.0 + 0.0im
                        for ig = 1:NG
                            for ic = 1:NC
                                local_trace += x[ic, ix, iy, iz, it, ig] * y[ic, ix, iy, iz, it, ig]
                            end
                        end
                        trace_sum[ix, iy, iz, it] = local_trace
                    end
                end
            end
        end

    return trace_sum
end

"""
    blockize(x::WF3x1, r::Int)

페르미온 벡터 `x`를 시공간 방향마다 `r`개씩 묶어 블록화한다.
NX, NY, NZ, NT 각각이 `r`로 나누어떨어져야 한다.

반환값: `Vector{WF3x1}` — 길이 `(NX÷r)*(NY÷r)*(NZ÷r)*(NT÷r)`.
각 원소는 원래와 같은 격자 크기의 WF3x1이며,
해당 블록에 속하는 사이트만 원래 값을 갖고 나머지는 0.
블록 순서는 (bx, by, bz, bt)를 row-major로 flatten한 것.
"""
function blockize(x::WF3x1, r::Int)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG

    @assert NX % r == 0 "NX=$NX 는 r=$r 로 나누어떨어져야 합니다."
    @assert NY % r == 0 "NY=$NY 는 r=$r 로 나누어떨어져야 합니다."
    @assert NZ % r == 0 "NZ=$NZ 는 r=$r 로 나누어떨어져야 합니다."
    @assert NT % r == 0 "NT=$NT 는 r=$r 로 나누어떨어져야 합니다."

    NBX = NX ÷ r
    NBY = NY ÷ r
    NBZ = NZ ÷ r
    NBT = NT ÷ r

    blocks = Vector{typeof(x)}(undef, NBX * NBY * NBZ * NBT)

    idx = 0
    @inbounds for bt = 1:NBT
        for bz = 1:NBZ
            for by = 1:NBY
                for bx = 1:NBX
                    idx += 1
                    blk = similar(x)   # 0으로 초기화된 WF3x1
                    for lt = 1:r
                        it = (bt - 1) * r + lt
                        for lz = 1:r
                            iz = (bz - 1) * r + lz
                            for ly = 1:r
                                iy = (by - 1) * r + ly
                                for lx = 1:r
                                    ix = (bx - 1) * r + lx
                                    for ig = 1:NG
                                        @simd for ic = 1:NC
                                            blk[ic, ix, iy, iz, it, ig] =
                                                x[ic, ix, iy, iz, it, ig]
                                        end
                                    end
                                end
                            end
                        end
                    end
                    blocks[idx] = blk
                end
            end
        end
    end

    return blocks
end

"""
    unblockize(blocks::Vector{WF3x1})

`blockize`의 역연산. 블록 벡터들을 모두 더해 원래 페르미온 벡터를 복원한다.
각 블록은 서로 겹치지 않는 사이트를 갖고 있으므로 단순 합산으로 복원 가능.
"""
function unblockize(blocks::Vector{T}) where {T<:WF3x1}
    f = similar(blocks[1])
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    NC, NG = f.NC, f.NG

    for blk in blocks
        @inbounds for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ig = 1:NG
                            @simd for ic = 1:NC
                                f[ic, ix, iy, iz, it, ig] +=
                                    blk[ic, ix, iy, iz, it, ig]
                            end
                        end
                    end
                end
            end
        end
    end

    return f
end
