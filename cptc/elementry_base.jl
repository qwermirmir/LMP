using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra


const WF3x1 = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}

abstract type Spin_vector end

struct spin_color_vector{T} <:Spin_vector 
    cvecs::Vector{T}
    # function spin_color_vector(U::Array{G,1},i::int) where{G<:AbstractGaugefields}
        

end


function spin_color_vector(U::Array{G,1}) where {G<:AbstractGaugefields}

    x1 = Initialize_pseudofermion_fields(U[1],"Wilson")

    spin_color_vector{typeof(x1)}(x1)

end



struct Spin_matrix <:Spin_vector
    sm::Matrix{ComplexF64}
    
end 

function Random_SpinM(matrix_type::String = "complex")
    if matrix_type == "complex"
        # 4x4 복소 랜덤 행렬 생성
        mat = randn(4,4) .+ im*randn(4,4)
    elseif matrix_type == "real"
        # 4x4 실수 랜덤 행렬 생성
        mat = randn(4,4)
        # 모든성분이 0인 행렬 생성 
    elseif matrix_type=="zero"
        mat = zeros(ComplexF64, 4,4)

    elseif matrix_type=="Init"
        mat = Matrix{ComplexF64}(I, 4, 4)
    
    else
        error("지원하지 않는 matrix_type 입니다. 'complex' 혹은 'real'을 사용하세요.")
    end
    
    return Spin_matrix(mat)
end




# getindex 정의
function Base.getindex(A::Spin_matrix, I...)
    return A.sm[I...]
end

function Base.setindex!(A::Spin_matrix, v, I...)
    A.sm[I...] = v
    return A
end



function Base.:+(a::WF3x1,b::WF3x1)

    y=deepcopy(b)



    axpby!(1,a,1,y)

    return y
end 



function Base.:-(a::WF3x1,b::WF3x1)

    y=deepcopy(b)



    axpby!(1,a,-1,y)

    return y
end 


function Base.:*(M::Spin_matrix,x::WF3x1)
    
    
    f=deepcopy(x)
    
    NX = f.NX
    NY = f.NY
    NZ = f.NZ
    NT = f.NT



    for ic = 1:NC
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        e1 = f[ic, ix, iy, iz, it, 1]
                        e2 = f[ic, ix, iy, iz, it, 2]
                        e3 = f[ic, ix, iy, iz, it, 3]
                        e4 = f[ic, ix, iy, iz, it, 4]
                        if (ic, ix, iy, iz, it) == (1, 1, 1, 1, 1)
                            println(e1,e2,e3,e4)
                        end

                        f[ic, ix, iy, iz, it, 1] =
                            M[1, 1] * e1 + M[1, 2] * e2 + M[1, 3] * e3 + M[1, 4] * e4
                        f[ic, ix, iy, iz, it, 2] =
                            M[2, 1] * e1 + M[2, 2] * e2 + M[2, 3] * e3 + M[2, 4] * e4
                        f[ic, ix, iy, iz, it, 3] =
                            M[3, 1] * e1 + M[3, 2] * e2 + M[3, 3] * e3 + M[3, 4] * e4
                        f[ic, ix, iy, iz, it, 4] =
                            M[4, 1] * e1 + M[4, 2] * e2 + M[4, 3] * e3 + M[4, 4] * e4
                    end
                end
            end
        end
    end
    return f
end

function Base.:*(b::Spin_matrix,a::Vector{WF3x1})




    for i in 1:length(a)
        a[i]=b*a[i]
    end

    return a
end













function Clifford()
    basis = Vector{Spin_matrix}()

    # 1) 첫번째 기저: 단위행렬(Identity)
    id4 = Random_SpinM("zero")
    for i in 1:4
        id4[i,i] = 1
    end
    push!(basis, id4)

    # 2) gamma1
    gamma1 = Random_SpinM("zero")
    gamma1[1,3] = 1
    gamma1[2,4] = 1
    gamma1[3,1] = 1
    gamma1[4,2] = 1
    push!(basis, gamma1)

    # 3) gamma2  (복소 항)
    gamma2 = Random_SpinM("zero")
    gamma2[1,4] = -im
    gamma2[2,3] =  im
    gamma2[3,2] =  im
    gamma2[4,1] = -im
    push!(basis, gamma2)

    # 4) gamma3
    gamma3 = Random_SpinM("zero")
    gamma3[1,3] =  1
    gamma3[2,4] = -1
    gamma3[3,1] = -1
    gamma3[4,2] =  1
    push!(basis, gamma3)

    # 5) gamma4 (혹은 gamma0)
    gamma4 = Random_SpinM("zero")
    gamma4[1,2] = 1
    gamma4[2,1] = 1
    gamma4[3,4] = -1
    gamma4[4,3] = -1
    push!(basis, gamma4)

    # 6) gamma5 (단순 예: 대각에 -1, -1, +1, +1 인 경우)
    gamma5 = Random_SpinM("zero")
    gamma5[1,1] = -1
    gamma5[2,2] = -1
    gamma5[3,3] =  1
    gamma5[4,4] =  1
    push!(basis, gamma5)

    # 7) pairwise products: gamma^mu gamma^nu (mu < nu)

    push!(basis,gamma1*gamma2)
    push!(basis,gamma1*gamma3)
    push!(basis,gamma1*gamma4)
    push!(basis,gamma2*gamma3)
    push!(basis,gamma2*gamma4)
    push!(basis,gamma3*gamma4)

    # 8) gamma^5 gamma^mu (4개)
    push!(basis, gamma5*gamma1)
    push!(basis, gamma5*gamma2)
    push!(basis, gamma5*gamma3)
    push!(basis, gamma5*gamma4)



    return basis
end


function Clifford(W::Spin_matrix)
    
    weight=Random_SpinM("zero")
    basis=Clifford()

    for i in 1:16
        r = div(i-1, 4) + 1
        c = mod(i-1, 4) + 1
        weight += W[r,c] * basis[i]
    end



    return weight
end 



function inner_product(x::WF3x1,f::WF3x1)
    
    k=0


    k=dot(x,f)


    return k
end




struct path_number
    f::Int64

    function path_number(x::Int64)
        # 허용되는 값인지 확인
        if x in (1, 2, 3, 4, -1, -2, -3, -4)
            return new(x)
        else
            error("path_number는 1,2,3,4,-1,-2,-3,-4 중 하나만 허용합니다.")
        end
    end
end

struct Path
    steps::Vector{path_number}
end

function Path(ints::Vector{Int})
    # ints: [1, 2, -4, ...]
    # 변환 -> path_number(...) 배열
    pn_array = [path_number(i) for i in ints]
    return Path(pn_array)
end

struct Paths
    various::Vector{Path}

    # Paths 생성자를 확장하여 가변 길이 인수를 받음
    function Paths(paths::Vector{Int64}...)
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end
end


function Onestep()
    dirs = (1, 2, 3, 4, -1, -2, -3, -4)
    # dirs를 Path로 매핑
    single_steps = [[d] for d in dirs]  # 각 값을 리스트로 감싸기
    return Paths(single_steps...)  # 가변 인자로 넘겨주기
end


struct Hopping_operator{G<:AbstractGaugefields}
    Gauge::Vector{G}
    P::Paths
end



function (hop::Hopping_operator)(x::WF3x1)


    shifts = Vector{WF3x1}()
    push!(shifts,deepcopy(x))

    for k in 1:length(hop.P.various)
        Ucopy = deepcopy(U)
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0
                # println(copyx[1,1,1,1,1,1])
                shift_x=shift_fermion(copyx , a)
                Uminus = shift_U(Ucopy[a], a)

                mul!(copyx, Uminus',shift_x)
                # println(a)
                # println(copyx[1,1,1,1,1,1])
                
            else
                # println(copyx[1,1,1,1,1,1])
                a = -a
                # Uplus = shift_U(Ucopy[a], -a)
                shift_x=shift_fermion(copyx, -a)
                # println(Ucopy[a][1,1,1,1,1,1])
                mul!(copyx, Ucopy[a], shift_x)
                # println(-a)
                # println(copyx[1,1,1,1,1,1])
                
            end
        end

        push!(shifts,copyx)
        println(copyx[1,1,1,1,1,1])
    end

    return shifts
end


function Generate_weight(x::Any; type="complex")
    W = Vector{Spin_matrix}()  # W 초기화

    if x isa Vector{WF3x1}
        for _ in x
            push!(W, Random_SpinM(type))
        end
    elseif x isa WF3x1
        push!(W, Random_SpinM(type))
    elseif x isa Hopping_operator
        for _ in 1:length(x.P.various)
            push!(W, Random_SpinM(type))
        end
    else
        throw(ArgumentError("지원되지 않는 x 타입: $(typeof(x))"))
    end

    return W  # 결과 반환
end





function PTC(W::Vector{Spin_matrix},hop::Hopping_operator,x::WF3x1) 

    result=deepcopy(x)



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


