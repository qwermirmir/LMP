
using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra


abstract type Spin_vector end

struct spin_color_vector{T} <:Spin_vector 
    cvecs::Vector{T}
    # function spin_color_vector(U::Array{G,1},i::int) where{G<:AbstractGaugefields}
        

end


function spin_color_vector(U::Array{G,1}) where {G<:AbstractGaugefields}

    x1 = Initialize_pseudofermion_fields(U[1],"Wilson")
    x2 = similar(x1)
    x3 = similar(x1)
    x4 = similar(x1)
    gauss_distribution_fermion!(x1)
    gauss_distribution_fermion!(x2)
    gauss_distribution_fermion!(x3)
    gauss_distribution_fermion!(x4)
    spin_color_vector{typeof(x1)}([x1,x2,x3,x4])

end 




function spin_color_vector(x1::T, x2::T, x3::T, x4::T) where T
    spin_color_vector{T}([x1, x2, x3, x4])
end

# getindex 메서드 정의
Base.getindex(x::spin_color_vector{T}, i::Int) where {T} = x.cvecs[i]

Base.setindex!(x::spin_color_vector{T}, v::T, i::Int) where {T} = (x.cvecs[i] = v)


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




function zero_spin!(x::spin_color_vector) 
    c1=similar(x[1])
    c2=similar(x[2])
    c3=similar(x[3])
    c4=similar(x[4])
    x[1]=c1
    x[2]=c2
    x[3]=c3
    x[4]=c4
    return nothing 

end



function hopping_operator(xout::T, U::Array{G,1}, x::T) where {T,G<:AbstractGaugefields}

    hopvector=[]
    for v in 1:4

    xminus = shift_fermion(x, -v)
    Uminus = shift_U(U[v], -v)

    mul!(xout,Uminus',xminus)

    

    push!(hopvector,xout)

    end 

    return hopvector

end


function hopping_operator( U::Array{G,1},x::spin_color_vector) where{G<:AbstractGaugefields}


    

    hopvector = Vector{spin_color_vector}()
    
    push!(hopvector,x)
    
    

    #처음엔 forward부터 계산한다.
    #H_{mu}\psi=U^{dagger}_{mu}(x-mu)\psi(x-mu)
    for v in 1:4 
        parts=[]
        for u in 1:4

        copyx=similar(x[u])
        Ushift=shift_U(U[v],-v)
        xshift=shift_fermion(x[u],-v)
        mul!(copyx,Ushift',xshift)
        push!(parts,copyx)
        end
        a=spin_color_vector(parts[1],parts[2],parts[3],parts[4])
        push!(hopvector,a)
    end


    return hopvector

end


function Base.:+(a::spin_color_vector,b::spin_color_vector)

    y=deepcopy(b)

    for i= 1:4

        axpby!(1,a[i],1,y[i])
    end
    return y
end 



function Base.:+(L::Array{G,1},K::Array{G,1}) where{G<:spin_color_vector}
    
    y = Vector{spin_color_vector}()


    for i=1:length(L)
        u=L[i]+K[i]
        push!(y,u)
    end
    return y

end    







function Base.:-(a::spin_color_vector,b::spin_color_vector)

    y=deepcopy(b)

    for i= 1:4

        axpby!(1,a[i],-1,y[i])
    end
    return y
end 

function Base.:-(L::Array{G,1},K::Array{G,1}) where{G<:spin_color_vector}
    
    y = Vector{spin_color_vector}()


    for i=1:length(L)
        b=L[i]-K[i]
        push!(y,b)
    end
    return y

end    

function Base.:*(A::Spin_matrix, B::Spin_matrix)
    # A.sm, B.sm 는 각각 4x4 ComplexF64 행렬
    return Spin_matrix(A.sm * B.sm)
end


function Base.:*(α::Number, W::Spin_matrix)
    return Spin_matrix(α .* W.sm)
end

function Base.:+(A::Spin_matrix,B::Spin_matrix)
    return Spin_matrix(A.sm+B.sm)

end


function Base.deepcopy_internal(f::spin_color_vector)
    
    x1=deepcopy(f[1])
    x2=deepcopy(f[2])
    x3=deepcopy(f[3])
    x4=deepcopy(f[4])

    return spin_color_vector(x1,x2,x3,x4)

end 



function Base.deepcopy_internal(f::Array{S,1}) where{S<:spin_color_vector}
    
    arr=[]
    b=length(f)
    for i=1:b

        push!(arr,f[i])

    end 

    return arr[]
end 

function zero_spin(f::spin_color_vector)


    NC=f[1].NC
    NX=f[1].NX
    NZ=f[1].NZ
    NY=f[1].NY
    NT=f[1].NT
    Nwing=2
    Dim=4



    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    x1= Initialize_pseudofermion_fields(U[1],"Wilson")
    x2= Initialize_pseudofermion_fields(U[1],"Wilson")
    x3= Initialize_pseudofermion_fields(U[1],"Wilson")
    x4= Initialize_pseudofermion_fields(U[1],"Wilson")



    spin=spin_color_vector(x1,x2,x3,x4)


    return spin 



end

function Base.:*(W::Spin_matrix,f::spin_color_vector) 
    
    spin=zero_spin(f)



    for i =1:4
        y=deepcopy(f)
        for j= 1:4
            spin[i]=axpby!(1,spin[i],W[i,j],y[j])
        end
    end
    return spin 
end



function Base.:*(W::Spin_matrix,v::Array{S,1}) where{S<:spin_color_vector}

    k=Vector{spin_color_vector}()

    for i=1:length(v)
        push!(k,W*v[i])
    end 

    return k

end 

function PTC(U::Array{G,1},f::spin_color_vector) where{G<:AbstractGaugefields}
    
    hop=hopping_operator(U,f)
    s=length(hop)
    W=Vector{Spin_matrix}()

    result=deepcopy(f)
    zero_spin(result)

    for i=1:s
        m=Random_SpinM("real")
        push!(W,m)
        hop[i]=m*hop[i]
    end

    for i=1:s
        result+=hop[i]
    end
    
    
    return result,W
end


function inner_product(x::spin_color_vector,f::spin_color_vector)
    
    k=0

    for i=1:4
        k+=dot(x[i],f[i])
    end

    return k
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

















