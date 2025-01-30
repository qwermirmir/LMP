using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra


const WF3x1 = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_wing{3,1}

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

function ters(x::WF3x1)
    return 2
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


