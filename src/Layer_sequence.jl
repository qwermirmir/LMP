module Layer_sequence





import ..Layer_types: Layer, CPTC_Layer, PTC_Layer,ActivationFunction,Identity,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer,minmaxNormLayer,zscoreNormLayer,RealAttention_Layer
import ..Layer_types: FullPTCAttention_Layer,Dirac_Layer,PrallelTransportAttention_Layer,siteweight_Layer,RayleighQuotient_Layer,PtcRayleighQuotient_Layer,HoppingRayleighQuotient_Layer
import ..Layer_types: Orthogonal_Layer,LossSumLayer,Ritz_Layer
import ..elementry_base: Hopping_operator, Spin_matrix, inner_product,g2vector_generator!,addmul!
import ..elementry_base: Clifford, Random_SpinM,norm2,mean,standardization,minmax,spincolortrace
import ..Optimizer: Opt, update!, Adam
import ..activation:  gradient, DeltaRule,DeltaRule!
import ..gpu: Matmul!,gpuaddmul!
import ..Layer_types: DenseLinearLayer, PTLayer,fermionfield_Layer,GramSchmidt_Layer
abstract type Operator end







# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Plots
using CUDA

# import LatticeDiracOperators: Dirac_Operators

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}
# const Dirac_operator =LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}
# const DagDirac_operator =LatticeDiracOperators.Dirac_operators.Adjoint_Wilson_operator{LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}


export forward_propagation, Lossfunction, train,  preD,Operator ,doubletrain,backward_propagation!




function Lossfunction(f::WF3x1,t::WF3x1)

    loss=f-t
    
    return inner_product(loss,loss)

end




function Lossfunction(f::Vector{WF3x1}, t::Vector{WF3x1})
    @assert length(f) == length(t)
    loss = 0.0
    for i in 1:length(f)
        diff = f[i] - t[i]
        loss += inner_product(diff, diff)
    end
    return loss / length(f)
end




function (layer::PTC_Layer)(x::WF3x1)
    result = similar(x)
    fields = layer.hop(x)  # Hopping operator를 적용한 field들
    c=similar(x)

    if length(layer.W) == 1
        # 가중치가 하나뿐인 경우 (모든 field에 같은 가중치를 곱함)
        for i in 1:length(fields)
            mul!(c, layer.W[1], fields[i])
            axpby!(1,c,1,result)
        end
    elseif length(layer.W) == length(fields)
        # 가중치 벡터와 field 벡터 크기가 같은 경우
        for i in 1:length(fields)
            mul!(c, layer.W[i], fields[i])
            axpby!(1,c,1,result)
        end
    else
        error("W의 길이가 1이거나 fields와 같아야 합니다.")
    end


    result = layer.activation(result)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end


    return result
end

function (layer::CPTC_Layer)(x::WF3x1)

    fields = layer.hop(x)  # Hopping operator를 적용한 field들
    cW=Clifford(layer.W)
    y=similar(x)
    c=similar(x)



    if length(layer.W) == 1
        # 가중치가 하나뿐인 경우 (모든 field에 같은 가중치를 곱함)
        for i in 1:length(fields)
            mul!(c, cW[1], fields[i])
            axpby!(1,c,1,y)
        end
    elseif length(layer.W) == length(fields)
        # 가중치 벡터와 field 벡터 크기가 같은 경우
        for i in 1:length(fields)
            mul!(c, cW[i], fields[i])
            axpby!(1,c,1,y)
        end
    else
        error("W의 길이가 1이거나 fields와 같아야 합니다.")
    end


    result =layer.activation(y)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end

    return result 
end


function  (layer::Attention_Layer)(x::WF3x1)


    QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)

    lW=length(QW)


    m=layer.nominator


    Hx=layer.V.hop(x)



    Q=similar(x)
    K=similar(x)
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        mul!(Q,QW[i],Hx[i])
        mul!(K,KW[i],Hx[i])

        S[i]=layer.activation(Matmul!(conj(Q),K)/m)
        # M[i]=Matmul!(conj(Q),K)/m
        # println("M[$(i)]: ", M[i])
        # println("S[$(i)]: ", S[i])

    
    end 

    for i in 1:lW
        mul!(S[i],S[i],VW[i])
    end




    z=similar(x)
    

    temp =similar(x)
    for i in 1:length(KW)
        mul!(temp,S[i],Hx[i])
        axpby!(1,temp,1,z)
    end



    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end 
    return z




end




function  (layer::test_Layer)(x::WF3x1)


    # QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)
    t=layer.test


    m=layer.nominator


    Hx=layer.V.hop(x)
    Ht=layer.V.hop(t)

    # if Q.W === nothing || K.W === nothing || V.W === nothing
    #     error("gradient() 호출 시 Q, K, V의 W가 정의되지 않았습니다.")
    # end

    Kx=KW*layer.K.hop(x)
    # Qx=KW*layer.Q.hop(x)
    # Vx=VW*layer.V.hop(x)

    M=similar(KW)
    S=similar(KW)
    for i in 1:length(KW)
        M[i]=Matmul!(conj(Kx[i]-t),Kx[i]-t)/m
        S[i]=layer.activation(M[i])
    end 

    Hx=layer.V.hop(x)
    # W=S*VW


    for i in 1:length(KW)
        mul!(S[i],S[i],VW[i])
    end


    z=similar(x)
    temp= similar(x)
    
    # for i in 1:length(KW)
    #     @inbounds z+=W[i]*Hx[i]
    # end

    for i in 1:length(KW)
        mul!(temp,S[i],Hx[i])
        axpby!(1,temp,1,z)
    end



    # if layer.residual == true 
    #     z=z+x
    # end 

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end

    return z

end


function (layer::FullPTCAttention_Layer)(x::WF3x1)


    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W




    lW=length(KW)


    m=layer.nominator


    Hx=layer.V.hop(x)


    # if Q.W === nothing || K.W === nothing || V.W === nothing
    #     error("gradient() 호출 시 Q, K, V의 W가 정의되지 않았습니다.")
    # end

    Kx=KW*layer.K.hop(x)
    Qx=QW*layer.Q.hop(x)


    M=similar(KW)
    S=similar(KW)
    for i in 1:lW
        M[i] = Matmul!(conj(Qx[i]), Kx[i]) / m
        S[i] = layer.activation(M[i])
        
    end
    

    Hx=layer.V.hop(x)
    W=S*VW
    z=similar(x)


    temp= similar(x)

    for i in 1:length(KW)
        mul!(temp,W[i],Hx[i])
        axpby!(1,temp,1,z)
    end

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end




    return z
end 

function (D::Dirac_Layer)(x::WF3x1)

    Dirac=D.activation
    y=Dirac*x
    return y
end 

function (layer::PrallelTransportAttention_Layer)(x::WF3x1)

    # QW=layer.Q.W
    KW=layer.K.W
    VW=layer.V.W

    lW=length(KW)
    m=layer.nominator
    b=layer.bias

    Kx=KW*layer.K.hop(x)
    # Qx=QW*layer.Q.hop(x)
    Hx=layer.V.hop(x)
    M=similar(KW)
    S=similar(KW)
    R=similar(KW)


    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m
        for a in 1:4
            R[i][a,a]=abs(M[i][a,a])
        end

    end 

    ratio = zeros(Float64, 4)           # ✅



    for u in 1:lW
        for a in 1:4
            ratio[a]+=R[u][a,a]
        end
    end

    for i in 1:lW
        for a in 1:4
            val = max(real(R[i][a,a] / ratio[a] - b),0)
            S[i][a,a]=val*M[i][a,a]
        end
    end

 

    # println("ratio: ", ratio[2])

    W=S*VW
    z=similar(x)




    temp= similar(x)

    for i in 1:length(KW)
        mul!(temp,W[i],Hx[i])
        axpby!(1,temp,1,z)
    end

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end



    return z
end 

function (layer::siteweight_Layer)(x::WF3x1)

    QW=layer.Q.W
    KW=layer.K.W
    VW=layer.V.W

    lW=length(KW)
    m=layer.nominator


    Hx=layer.V.hop(x)

    Kx=KW*layer.K.hop(x)
    Qx=QW*layer.Q.hop(x)
    
    Hx=layer.V.hop(x)
    Vx=VW*layer.V.hop(x)

    result = similar(x)
    for p in 1:lW
        a=similar(x)
        S=Vector{Array{ComplexF64, 4}}(undef, lW)
        for q in 1:lW
            S[q] = spincolortrace(conj(Qx[p]), Kx[q]) / m
            # n=norm(vec(S[q]))
            # S[q]=S[q]/n
            a += S[q] * Vx[q]

        end
        result += a
    end 




    if layer.residual == true 
        # result=result+x
        axpby!(1,x,1,result)
    end

    return result
end


function (layer::HoppingRayleighQuotient_Layer)(D::Dirac_operator)


    x=layer.x

    Hx=layer.hop(x)

    y=similar(x)
    for i in 1:length(Hx)
        axpby!(1,Hx[i],1,y)
    end
    u=D*y
    
    rq=inner_product(u,u)/inner_product(y,y)
    
    return rq
end


# function (layer::DenseLinearLayer)(x::Vector{WF3x1CPU}) 
#     Out=layer.Out
#     W=layer.W


#     @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."

#     result = Vector{WF3x1}(undef, Out)

#     for k in 1:length(x)
#         @inbounds for i in 1:Out
#             result[i]=W[i+(k-1)*Out]*x[k]
#             result[i] = layer.activation(result[i])
#         end
#     end

#     return result
    
# end


function (layer::DenseLinearLayer)(x::Vector{WF3x1GPU}) 
    Out = layer.Out
    W   = layer.W

    @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."
    result = Vector{WF3x1}(undef, Out)
    yi = deepcopy(x[1])

    for k in 1:length(x)
        @inbounds for i in 1:Out
                # yi용 버퍼 1개 생성
            mul!(yi, W[i+(k-1)*Out], x[k])            # yi = W[i]*x 를 "할당 없이" 계산
            result[i] += layer.activation(yi)  # activation이 새로 만들면 여기서 또 alloc
        end
    end 
    return result
end


function (layer::DenseLinearLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out = layer.Out
    W   = layer.W

    @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."
    result = Vector{WF3x1}(undef, Out)
    temp = similar(x[1])
    for i in 1:Out
        result[i] = deepcopy(temp)  # 초기화 (GPU에서는 이 부분이 병목이 될 수 있음)
    end
    yi = deepcopy(x[1])

    for k in 1:length(x)
        @inbounds for i in 1:Out
                # yi용 버퍼 1개 생성
            mul!(yi, W[i+(k-1)*Out], x[k])            # yi = W[i]*x 를 "할당 없이" 계산
            result[i] += layer.activation(yi)  # activation이 새로 만들면 여기서 또 alloc
        end
    end 
    return result
    
end





# function (layer::DenseLinearLayer)(x::WF3x1) 
#     Out = layer.Out
#     W   = layer.W

#     result = Vector{WF3x1}(undef, Out)
#     yi = deepcopy(x)   

    
#     @inbounds for i in 1:Out
#            # yi용 버퍼 1개 생성
#         mul!(yi, W[i], x)            # yi = W[i]*x 를 "할당 없이" 계산
#         result[i] = layer.activation(yi)  # activation이 새로 만들면 여기서 또 alloc
#     end

#     return result
# end


using Base.Threads

function (layer::PTLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out   = layer.Out
    W     = layer.W
    paths = layer.paths


    npaths = length(paths)

    @assert npaths == length(x) "입력 x의 길이는 paths의 길이와 같아야 합니다."
    @assert length(W) == Out * npaths "W의 길이는 Out*paths이어야 합니다."

    # 1) hop(x[i])[1]을 미리 전부 계산 (단일 스레드)
    hop = Hopping_operator(layer.U, paths[1])
    # y=deepcopy(x[1])
    hopped = Vector{WF3x1}(undef, npaths)
    @inbounds for i in 1:npaths
        hop.P = paths[i]
        # hopped[i] = hop(x[i])[1]      
        hi = hop(x[i])[1]
        hopped[i] = deepcopy(hi)        
    end

    # 2) 결과 벡터 초기화
    temp = similar(x[1])
    result = Vector{WF3x1}(undef, Out)
    for j in 1:Out
        result[j] = deepcopy(temp)
    end

    # 3) Out 방향으로 병렬화
    if x[1] isa WF3x1CPU
        @threads for j in 1:Out
            rj = result[j]    # 이 스레드가 책임지는 출력

            @inbounds for i in 1:npaths
                w = W[(i-1)*Out + j]
                addmul!(rj, w, hopped[i])
            end
        result[j] = rj   
        end
    else
        for j in 1:Out
            rj = result[j]   

            @inbounds for i in 1:npaths
                gpuaddmul!(rj, W[(i-1)*Out + j], hopped[i])
            end
            result[j] = rj   
        end
    end

    return result
end

function (layer::fermionfield_Layer)(x::Any)



    # x=layer.x
    # result = layer.activation(x)


    # if x isa Vector{WF3x1}
    #     result = Vector{WF3x1}(undef, layer.In)
    #     for i in 1:layer.In
    #         result[i] = layer.activation(x[i])
    #     end
    # else 
    #     result = Vector{WF3x1}(undef, layer.In)
    #     for i in 1:layer.In
    #         result[i] = layer.activation(x)
    #     end
    # end


    return layer.x
end









function (layer::NormLayer)(x::Vector{WF3x1})

    result = Vector{WF3x1}(undef, length(x))
    for i in 1:length(x)
        result[i] = layer.alpha*(x[i]/sqrt(norm2(x[i])))
    end 

    return result
end

function (layer::complexNormLayer)(x::WF3x1)


    m=mean(x)
    x=x-m

    return x
end

function (layer::zscoreNormLayer)(x::WF3x1)
    x=standardization(x)

    return x
end 

function (layer::minmaxNormLayer)(x::WF3x1)
    
    
    min,max=minmax(x)

    x=x-min
    x=x/(max-min)

    return x
end












function (layer::RayleighQuotient_Layer)(D::Dirac_operator)
    x=layer.x
    y=D*x
    rq=norm2(y)/norm2(x)

    return rq
end


function (layer::PtcRayleighQuotient_Layer)(x::Vector{WF3x1})

    D=layer.D
    rq=0.0
    Dx=similar(x[1])

    for i in 1:length(x)

    

        mul!(Dx,D,x[i])

    # Dx=D*x[1]
        rq+=inner_product(Dx,Dx)/inner_product(x[i],x[i])

    end 
    return rq
end

# function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})

#     l=length(x)
#     # V=x
#     A= Matrix{ComplexF64}(undef, l, l)
#     B= Matrix{ComplexF64}(undef, l, l)
#     D=layer.D
#     # DDAGD를 쓸수가 없음 아직 LatticeDiracOperators에 구현이 안되어있음


#     for i in 1:l
#         for j in 1:l
#             y=similar(x[j])
#             mul!(y,D,x[j])
#             u=similar(y)
#             mul!(u,D,x[i])
#             A[i,j]=inner_product(u,y)
#             B[i,j]=inner_product(x[i],x[j])
#         end
#     end
#     # println(A)
#     # println(B)

#     evals, evecs = eigen(A, B)

#     lambda=0
#     for i in 1:length(evals)
#         lambda+=real(evals[i])
#     end

#     return lambda
#     # return A,B
# end
using LinearAlgebra
using Base.Threads

function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)


    # 1) Dx[i] = D * x[i] 를 i당 1번만 계산
    T = typeof(x[1])
    Dx = Vector{T}(undef, l)
    @inbounds for i in 1:l
        Dx[i] = similar(x[i])
        mul!(Dx[i], layer.D, x[i])   # mul!이 overwrite라고 가정 (누적이면 먼저 0으로 초기화 필요)
    end

    # 2) Gram matrices: A_ij=<Dx_i, Dx_j>, B_ij=<x_i, x_j>
    A = Matrix{ComplexF64}(undef, l, l)
    B = Matrix{ComplexF64}(undef, l, l)

    # CPU면 threads로 outer i를 병렬화 가능 (GPU x일 때는 inner_product가 sync를 많이 걸 수 있어 비추)
    @inbounds for i in 1:l
        # diagonal
        bii = inner_product(x[i],  x[i])
        aii = inner_product(Dx[i], Dx[i])
        B[i,i] = bii
        A[i,i] = aii

        # upper triangle만 계산하고 Hermitian으로 채우기
        for j in i+1:l
            bij = inner_product(x[i],  x[j])
            aij = inner_product(Dx[i], Dx[j])

            B[i,j] = bij
            B[j,i] = conj(bij)

            A[i,j] = aij
            A[j,i] = conj(aij)
        end
    end

    # 3) generalized eigen (Hermitian으로 감싸면 내부 알고리즘이 더 맞고 조금 더 빠름)
    F = eigen(Hermitian(A), Hermitian(B))

    return real(sum(F.values))
end




function (layer::Orthogonal_Layer)(x::Vector{WF3x1})

    l=length(x)
    cl=(l-1)*l/2
    a=layer.coeff

    result=0.0
    
    for i in 2:l
        for j in 1:i-1
            c=inner_product(x[j],x[i])
            result+=c'*c
        end
    end 


    return a*result
end 


function (layer::GramSchmidt_Layer)(x::Vector{WF3x1})
    V = layer.x                 # V[j] 들은 서로 직교라고 가정
    l = length(x)


    
    result = Vector{WF3x1}(undef, l)
    temp   = similar(x[1])

    @inbounds for i in 1:l
        y = deepcopy(x[i])      # y = x_i
        for j in 1:length(V)
            denom = inner_product(V[j], V[j])
            coeff = inner_product(V[j], y) / denom
            axpby!(-coeff, V[j], 1, y)   # y -= coeff * V[j]

        end
        result[i] = y
    end

    return result
end




# function (layer::L2Norm_Layer)(x::Any)

#     if x isa Vector{WF3x1}

#         return norm2(x[1]-layer.target)
#     else 
#         return norm2(x - layer.target)
#     end
# end


function (L::LossSumLayer)(x::Vector{WF3x1})
    return L.w1 * L.l1(x) + L.w2 * L.l2(x)
end


function Base.:+(a::A, b::B) where {A<:Layer, B<:Layer}
    return LossSumLayer(a, b)
end


function (layers::Vector{T})(x::Any) where T<:Layer
    result = x
    for layer in layers
        result = layer(result)
    end
    return result
end











function forward_propagation(layers::Vector{T}, input::Any, target::Any) where {T<:Layer}
    L = length(layers)

    # u는 "마지막 레이어 입력까지" 담습니다. 즉 길이 = L
    # u[1] = 입력, u[2] = 1번째 레이어 출력, ..., u[L] = (L-1)번째 레이어 출력


    
    let last = layers[L]
        if hasfield(typeof(last), :target)   # 또는: hasproperty(last, :target)
            # last가 mutable이면 아래 대입이 됩니다.
            last.target = target             # (= setproperty!(last, :target, target))
        end
    end

    u = Vector{Any}(undef, L)
    u[1] = [input]

    # 1..L-1까지는 WF3x1 → WF3x1
    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    # 마지막 레이어는 WF3x1 → Float64
    f = layers[L](u[L])  # 스칼라

    return u, f
end


export backward_propagation




function backward_propagation(layers::Vector{T},  u::Vector{Any}, batch::Int) where T<:Layer
    @assert length(u) == length(layers) 
    @assert all(isassigned(u, j) for j in 1:length(u))  # u가 전부 채워졌는지 안전 확인(선택)

    delta = Vector{Any}(undef, length(layers)-1)


    # 출력층 델타: u[end] = u[L+1] (마지막 레이어 출력)
    delta[1] = DeltaRule(layers[end], layers[end].activation, u[end], batch)

    if length(layers) > 2
        for i in 1:length(layers)-2
            # l = L - i  에 해당. 활성화는 반드시 u[l+1]을 써야 함!
            delta[1+i] = DeltaRule(
                layers[length(layers) - i],
                layers[length(layers) - i].activation,
                u[length(layers) - i ],   # ← 여기 +1 이 핵심 수정
                delta[i]
            )
        end
    end

    @assert length(delta) == length(layers) - 1

    return delta
end

function backward_propagation!(layers::Vector{T},  u::Vector{Any}, batch::Int, delta::Vector{Any}) where T<:Layer
    @assert length(u) == length(layers) 
    @assert all(isassigned(u, j) for j in 1:length(u))  # u가 전부 채워졌는지 안전 확인(선택)

    # delta = Vector{Any}(undef, length(layers)-1)
    @assert length(delta) == length(layers) "delta 벡터는 layers보다 하나 작아야 합니다. 현재 길이: $(length(delta))"


    # 출력층 델타: u[end] = u[L+1] (마지막 레이어 출력)
    @time DeltaRule!(layers[end], layers[end].activation, u[end], delta[end] ,batch)

    @time if length(layers) > 2
        for i in 1:length(layers)-2
            # l = L - i  에 해당. 활성화는 반드시 u[l+1]을 써야 함!
           DeltaRule!(
                layers[length(layers) - i],
                layers[length(layers) - i].activation,
                u[length(layers) - i ],   # ← 여기 +1 이 핵심 수정
                delta[length(layers)-i+1],
                delta[length(layers)-i]
            )
        end
    end



    return delta
end

using Plots
using Base.Iterators
using DisplayAs
using PyPlot




function train(opt::Opt, layers::Vector{T}, input::WF3x1, target::WF3x1 , Dirac_operator ; epochs=100, batch=1,precond=nothing , def=false) where {T<:Layer}
    depth = length(layers)
    loss_history = Float64[]  # 손실 값 저장할 벡터

    # Jupyter에서 그래프를 실시간으로 업데이트하기 위한 설정
    fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)  # 초기 그래프

    # display(fig)  # Jupyter에서 그래프 창 한 번만 표시


    function cosine_annealing_lr(epoch, α_max, α_min, T_max)
        t = epoch % T_max
        return α_min + 0.5 * (α_max - α_min) * (1 + cos(π * t / T_max))
    end


    M=precond


    γ=Clifford()


    D = Dirac_operator


    y = similar(input)
    x= similar(input)
    # target=deepcopy(input)

    for epoch in 1:epochs

        current_loss = 0.0
        temp=Vector{ComplexF64}(undef, batch)
        d=Vector{Any}(undef, depth)
        # for i in 1:depth
        #     if layers[i].W === nothing
        #         d[i]=nothing
        #         continue
        #     end
        #     d[i]= similar(layers[i].W)
        # end 

        for b in 1:batch

            # g2vector_generator!(input)
            gauss_distribution_fermion!(input)

            # target=deepcopy(input)/norm(input)

            
            #deflation
            if def== true 
                k=similar(input)
                gmres(input,D,k,maxsteps=1,restart=5)
            end

            if D isa Vector{T}
                y=D(input)
            else 
                mul!(y,D,input)
            end
            # mul!(x,D,y)


            #left
            den=norm(y) 

            #Right
            # den=norm(input)

            # target=input
            target=(input)/den
            if M != nothing
                target=target - M(y/den)
            end
            #left
            input=y/den 
            # input=(y)/den
            # input=x

            # # #Right
            # input=input/den
            
            # input=y-input
            # target=similar(input)


            u,f = forward_propagation(layers, input , target)
            temp[b]=f

            delta = backward_propagation(layers, target, u, batch)

            for i in 1:depth-1
                if b == 1
                    d[i] = gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i ],γ)
                
                    # if d[i] === nothing
                    #     continue  # NormLayer 등은 업데이트 생략
                    # end
                    d[i] += gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i ],γ)
                end
                # d[i] += gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i + 1]/batch,γ)

            end





      

        end

        # Gradient 계산, grad에 모든 층의 gradient 저장
         @time for i in 1:depth-1
            # d = gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i + 1],γ)
            
            # if d === nothing
            #     continue  # NormLayer 등은 업데이트 생략
            # end
        

        # Optimizer로 가중치 업데이트
            update!(layers[i], opt, d[i])

        end

        # 현재 손실 계산 후 저장
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



        # 기존 그래프를 지우고 다시 그리기
        
        println("Epoch: $epoch, Loss: $current_loss")  # 로그 출력



    end
    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end







function train(opt::Opt, layers::Vector{T} , config ,params ; epochs=100, batch=1 ) where {T<:Layer}

    depth = length(layers)

    loss_history = Float64[]  # 손실 값 저장할 벡터


        fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)  # 초기 그래프
    γ=Clifford()

    x = Initialize_pseudofermion_fields(config[1][1],"Wilson")
    D= Dirac_operator(config[1],x,params)
    delta ,f = forward_propagation(layers, D , D)

    for epoch in 1:epochs
        k=mod(epoch-1,length(config))+1
        D= Dirac_operator(config[k],x,params)
        for i in 1:depth
            if layers[i] isa PTLayer
                layers[i].U= config[k]
            end 
        end
        if layers[end] isa PtcRayleighQuotient_Layer
            layers[end].D=D
        elseif  layers[end] isa LossSumLayer
            if layers[end].l1 isa PtcRayleighQuotient_Layer
                layers[end].l1.D=D
            end
            if layers[end].l2 isa PtcRayleighQuotient_Layer
                layers[end].l2.D=D
            end
        end
        if layers[end] isa Ritz_Layer
            # DdagD=DdagD_operator(config[k],x,params)
            # layers[end].D=DdagD
            D= Dirac_operator(config[k],x,params)
            layers[end].D=D
        end
        temp=Vector{ComplexF64}(undef, batch)
        d=Vector{Any}(undef, depth)
        fill!(d, nothing)
        for b in 1:batch
             u,f = forward_propagation(layers, D , D)
            temp[b]=f
            backward_propagation!(layers, u, batch, delta)



            for i in 1:depth-1
                gi = gradient(layers[i], layers[i].activation, u[i], delta[i+1], γ)

                gi === nothing && continue

                if d[i] === nothing
                    d[i] = gi
                else
                    d[i] += gi
                end
            end
        end


        # Gradient 계산, grad에 모든 층의 gradient 저장
         for i in 1:depth-1
        # Optimizer로 가중치 업데이트
            d[i] === nothing && continue
           update!(layers[i], opt, d[i])
        end

        # 현재 손실 계산 후 저장
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



        # 기존 그래프를 지우고 다시 그리기

        println("Epoch: $(epoch), Loss: $current_loss")  # 로그 출력
    end


    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end








end # module Layer_sequence