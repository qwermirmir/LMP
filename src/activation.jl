module activation


# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using CUDA
using Base.Threads
  # 활성화 함수 추상 타입


import ..elementry_base: Spin_matrix, inner_product, Clifford, reverse_paths,Hopping_operator,norm2,mean,minmax,spincolortrace,addmul!
import ..Layer_types: CPTC_Layer, PTC_Layer, Layer,ActivationFunction,Identity ,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer,minmaxNormLayer,zscoreNormLayer,siteweight_Layer
import ..Layer_types: FullPTCAttention_Layer,Dirac_Layer,PrallelTransportAttention_Layer,ModRelu,RayleighQuotient_Layer,PtcRayleighQuotient_Layer
import ..Layer_types: HoppingRayleighQuotient_Layer,DenseLinearLayer,PTLayer,fermionfield_Layer,LossSumLayer,Orthogonal_Layer,Ritz_Layer,GramSchmidt_Layer
import ..gpu: Matmul!,Spin_matrix_GPU,gpuaddmul!,zero_wf!,conj!

export gradient, DeltaRule,DeltaRule!
export Identity, ReLU, Sigmoid, Tanh

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}





function gradient(layer::CPTC_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})
    if layer.W === nothing
        error("gradient() 호출 시 layer.W가 정의되지 않았습니다.")
    end


    # cd = conj(d)

    lW = length(layer.W) 

    s = layer.hop(z)
    grad = deepcopy(layer.W)

    for i in 1:lW
        for u in 1:4
            for v in 1:4
                grad[i][u, v] = inner_product(conj(d),γ[4*(u-1)+v]*s[i])
            end
        end
    end 

    # println(inner_product(d,z))
    # println(inner_product(cd,z))

    # 이 부분에선 복소수 함수이므로, 라이프니츠법칙에 따라야한다. 

    grad=grad+conj(grad)

    return grad/2
end


function DeltaRule(layer::CPTC_Layer, ::Identity,z::WF3x1, d::WF3x1)
    hop = layer.hop
    W = Clifford(layer.W)
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    dH=invhop(conj(d)) 

    
    zW=similar(W)
    for i in 1:length(W)
        for a in 1:4
            for b in 1:4
                zW[i][a,b]=W[i][b,a]
            end
        end
    end

    result=similar(d)
    for i in 1:length(W)
        result+=zW[i]*conj(dH[i])
    end

    if layer.residual == true
        result=result+d
    end


    return result
end 



function gradient(layer::PTC_Layer,::Identity, z::WF3x1, d::WF3x1,γ::Vector{Spin_matrix} )

    if layer.W === nothing
        error("gradient() 호출 시 layer.W가 정의되지 않았습니다.")
    end

    hop = layer.hop
    if hop === nothing
        error("gradient() 호출 시 layer.hop이 정의되지 않았습니다.")
    end

    W = layer.W
    lW = length(W)

    s = hop(z)
    grad = deepcopy(W)

    for i in 1:lW
        grad[i]=Matmul!(d,s[i])
    end 

    return grad



end



function DeltaRule(layer::PTC_Layer, ::Identity,z::WF3x1, d::WF3x1)


    hop = layer.hop
    cW = adjoint(layer.W)
    lW = length(cW)


    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)
    cd = conj(d) # 여긴 문제가 없음
    cL = PTC_Layer(invhop,cW)

    result = cL(cd) # 여긴 문제가 없음
    result = conj(result) # 여긴 문제가 없음 

    
    if layer.residual == true
        # result=result+d
        axpby!(1,d,1,result)
    end


    return result
end


function gradient(layer::Attention_Layer, ::Softmax, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})
    
    




    QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)

    lW=length(QW)

    m=layer.nominator


    Hx=layer.V.hop(z)


    Kx=KW*layer.K.hop(z)
    Qx=QW*layer.Q.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 
    

    S=similar(QW)
    for i in 1:lW
        S[i]=layer.activation(Matmul!(conj(Qx[i]),Kx[i])/m)
    end 

    PMK = Vector{Vector{Spin_matrix}}(undef, lW)

    for i in 1:lW
        p = Vector{Spin_matrix}(undef, 16)
        for l in 1:16
            p[l] = Matmul!(conj(Qx[i]), γ[l]*Hx[i])/m
        end
        PMK[i] = p   # ✅ push! 대신 직접 할당
    end
    
    PMQ = Vector{Vector{Spin_matrix}}(undef, lW)

    for i in 1:lW
        p = Vector{Spin_matrix}(undef, 16)
        for l in 1:16
            p[l] = Matmul!(conj(γ[l]*Hx[i]), Kx[i])/m
        end
        PMQ[i] = p
    end
    





    
    #-------------------------------------------------------------------------------------------------------
    # Q의 graident를 계산하는 부분 

    grad=similar(QW)
    grad1=similar(QW)
    
    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        grad[i][a,b]+=PMQ[i][4*(a-1)+b][c,d]*S[i][c,d]*eta[i][d,c]
                
                    end
                end 
            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        for f in 1:4
                            grad1[i][a,b]+=PMQ[i][4*(a-1)+b][c,d]*S[i][c,f]*S[i][c,d]*eta[i][f,c]
                        end 
                    end
                end
            end
        end
    end 

    gradQ=grad-grad1
    gradQ=gradQ+conj(gradQ)
    gradQ=gradQ/2


#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(QW)
    grad1=similar(QW)




    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        grad[i][a,b]+=PMK[i][4*(a-1)+b][c,d]*S[i][c,d]*eta[i][d,c]
                
                    end
                end 
            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        for f in 1:4
                            grad1[i][a,b]+=PMK[i][4*(a-1)+b][c,d]*S[i][c,f]*S[i][c,d]*eta[i][f,c]
                        end 
                    end
                end
            end
        end
    end 


    gradK=grad-grad1
    gradK=gradK+conj(gradK)
    gradK=gradK/2



#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    grad=similar(QW)



    for i in 1:lW
        for u in 1:4
            for v in 1:4
                grad[i][u, v] = inner_product(conj(d),S[i]*γ[4*(u-1)+v]*Hx[i])
            end
        end
    end 



    gradV=grad/2
    gradV=gradV+conj(gradV)


    # gradV=similar(gradQ)

    gradlist=[gradQ,gradK,gradV]




    return gradlist


end 






function DeltaRule(layer::Attention_Layer, ::Softmax, z::WF3x1, d::WF3x1)
    Q = layer.Q
    K = layer.K
    V = layer.V



    QW = deepcopy(Q.W)
    KW = deepcopy(K.W)
    VW = deepcopy(V.W)

    QW=Clifford(QW)
    KW=Clifford(KW)
    VW=Clifford(VW)

    lW=length(QW)

    m=layer.nominator


    Hx=layer.V.hop(z)



    Qx=QW*layer.Q.hop(z)
    Kx=KW*layer.K.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 

    bareta = deepcopy(VW)
    for i in 1:lW
        bareta[i]=Matmul!(conj(Vx[i]),conj(d))
    end




    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]),Kx[i])/m
        S[i]=layer.activation(M[i])
    end 

  



    hop = layer.Q.hop
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)
    QH = [similar(Hx[i]) for i in 1:lW]
    KH = [similar(Hx[i]) for i in 1:lW]
    for i in 1:lW
        # QH[i]=QW[i]*z
        # KH[i]=KW[i]*z
        mul!(QH[i],QW[i],z)
        mul!(KH[i],KW[i],z)
    end
    dH=invhop(conj(d))

    result=similar(d)


    #--------------------------------------------------------------------------------------------------------




    Qz1 = similar(QW)
    Qz2 = similar(QW)


    for i in 1:length(QW)
        for a in 1:4,  c in 1:4, d in 1:4
            Qz1[i][a,c] += QW[i][d,a]*(adjoint(S[i])[c,d])*bareta[i][c,d]
        end
    end

    for i in 1:length(QW)
        for a in 1:4, c in 1:4, e in 1:4 , d in 1:4
            Qz2[i][a,c] +=  QW[i][d,a]*(adjoint(S[i])[e,d])*(adjoint(S[i])[c,d]) * bareta[i][e,d]
        end
    end







    Qz=Qz1-Qz2


    for i in 1:length(QW)
        # result+=Qz[i]*conj(KW[i]*z)/m
        axpby!(1,Qz[i]*conj(KW[i]*z)/m,1,result)
    end


        #--------------------------------------------------------------------------------------------------------
    # K의 delta를 계산하는 부분


    #첫번째항

    # Kz 계산 (정확한 버전)
    Kz1 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, c in 1:4 , m in 1:4
            Kz1[i][a,c] +=KW[i][m,a]* S[i][c,m]*eta[i][m,c]
        end
    end

    Kz2 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, d in 1:4, c in 1:4 , m in 1:4
            Kz2[i][a,c] +=KW[i][d,a]*S[i][c,m]*S[i][c,d]*eta[i][m,c]
        end
    end
    Kz = Kz1 - Kz2

    


    # # 최종 delta 계산
    for i in 1:length(KW)
        axpby!(1,Kz[i]*conj(QW[i]*z)/m,1,result)
    end







    #---------------------------------------------------------------------------------------------------------------------




    Vz=similar(VW)

    for i in 1:lW
        for a in 1:4
            for c in 1:4
                for e in 1:4
                    Vz[i][a,c]+=VW[i][e,a]*S[i][c,e]
                end
            end
        end
    end




    for i in 1:lW
        # result+=Vz[i]*conj(dH[i])
        axpby!(1,Vz[i]*conj(dH[i]),1,result)
    end    




    if layer.residual == true
        # result=result+d
        axpby!(1,d,1,result)
    end


    return result

end



function gradient(layer::test_Layer, ::Softmax, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})
    t=layer.test
    


    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W

    QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)

    lW=length(QW)



    m=layer.nominator


    Hx=layer.V.hop(z)


    Kx=KW*layer.K.hop(z)
    Qx=KW*layer.Q.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 
    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]-t),Kx[i]-t)/m
        S[i]=layer.activation(M[i])
    end 



    if QW[1] isa Spin_matrix


        PMK = Vector{Vector{Spin_matrix}}(undef, lW)

        for i in 1:lW
            p = Vector{Spin_matrix}(undef, 16)
            for l in 1:16
                p[l] =Matmul!(conj(Qx[i]-t), γ[l]*Hx[i])/m
            end
            PMK[i] = p   # ✅ push! 대신 직접 할당
        end
        
        PMQ = Vector{Vector{Spin_matrix}}(undef, lW)

        for i in 1:lW
            p = Vector{Spin_matrix}(undef, 16)
            for l in 1:16
                p[l] =Matmul!(conj(γ[l]*Hx[i]), Kx[i]-t)/m
            end
            PMQ[i] = p
        end
    
    else 

# 추천 코드 ✅
        γ = [Spin_matrix_GPU(g.sm) for g in γ]

        PMK = Vector{Vector{Spin_matrix_GPU}}(undef, lW)



                
        for i in 1:lW
            p = Vector{Spin_matrix_GPU}(undef, 16)
            for l in 1:16
                p[l] =Matmul!(conj(Qx[i]-t), γ[l]*Hx[i])/m
            end
            PMK[i] = p   # ✅ push! 대신 직접 할당
        end
        
        PMQ = Vector{Vector{Spin_matrix_GPU}}(undef, lW)

        for i in 1:lW
            p = Vector{Spin_matrix_GPU}(undef, 16)
            for l in 1:16
                p[l] =Matmul!(conj(γ[l]*Hx[i]), Kx[i]-t)/m
            end
            PMQ[i] = p
        end

    end
    
    #-------------------------------------------------------------------------------------------------------
    # Q의 graident를 계산하는 부분 
    # PM= deepcopy(QW)

    grad=similar(QW)
    grad1=similar(QW)
    
    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        @inbounds grad[i][a,b]+=PMQ[i][4*(a-1)+b][c,d]*S[i][c,d]*eta[i][d,c]
                
                    end
                end 
            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        for f in 1:4
                            @inbounds grad1[i][a,b]+=PMQ[i][4*(a-1)+b][c,d]*S[i][c,f]*S[i][c,d]*eta[i][f,c]
                        end 
                    end
                end
            end
        end
    end 

    gradQ=grad-grad1
    gradQ=gradQ+conj(gradQ)
    gradQ=gradQ/2


#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(QW)
    grad1=similar(QW)




    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        grad[i][a,b]+=PMK[i][4*(a-1)+b][c,d]*S[i][c,d]*eta[i][d,c]
                
                    end
                end 
            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for c in 1:4
                    for d in 1:4
                        for f in 1:4
                            grad1[i][a,b]+=PMK[i][4*(a-1)+b][c,d]*S[i][c,f]*S[i][c,d]*eta[i][f,c]
                        end 
                    end
                end
            end
        end
    end 


    gradK=grad-grad1
    gradK=gradK+conj(gradK)
    gradK=gradK/2

    gradK=gradK+gradQ



#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    grad=similar(QW)



    for i in 1:lW
        for u in 1:4
            for v in 1:4
                grad[i][u, v] = inner_product(conj(d),S[i]*γ[4*(u-1)+v]*Hx[i])
            end
        end
    end 



    gradV=grad/2


    gradV=gradV+conj(gradV)

    gradV=similar(gradQ)

    gradlist=[gradK,gradV]


    return gradlist


end 









function DeltaRule(layer::test_Layer, ::Softmax, z::WF3x1, d::WF3x1)
    Q = layer.Q
    K = layer.K
    V = layer.V
    t=layer.test


    QW = deepcopy(Q.W)
    KW = deepcopy(K.W)
    VW = deepcopy(V.W)

    QW=Clifford(QW)
    KW=Clifford(KW)
    VW=Clifford(VW)

    lW=length(QW)

    m=layer.nominator


    Hx=layer.V.hop(z)
    Ht=layer.V.hop(t)

    if Q.W === nothing || K.W === nothing || V.W === nothing
        error("gradient() 호출 시 Q, K, V의 W가 정의되지 않았습니다.")
    end

    Qx=KW*layer.Q.hop(z)
    Kx=KW*layer.K.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 

    bareta = deepcopy(VW)
    for i in 1:lW
        bareta[i]=Matmul!(conj(Vx[i]),conj(d))
    end

    
    # println(eta[1])
    # println(bareta[1])


    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]-t),Kx[i]-t)/m
        S[i]=layer.activation(M[i])
    end 

  



    hop = layer.Q.hop
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    QH=similar(Hx)
    KH=similar(Hx)
    for i in 1:lW
        KH[i]=invhop(Kx[i]-t)[i]
        # QH[i]=QW[i]*z
        # KH[i]=KW[i]*z
    end
    dH=invhop(conj(d))

    result=similar(d)


    #--------------------------------------------------------------------------------------------------------




    Qz1 = similar(QW)
    Qz2 = similar(QW)


    for i in 1:length(QW)
        for a in 1:4,  c in 1:4, d in 1:4
            Qz1[i][a,c] += KW[i][d,a]*(adjoint(S[i])[c,d])*bareta[i][c,d]
        end
    end

    for i in 1:length(QW)
        for a in 1:4, c in 1:4, e in 1:4 , d in 1:4
            Qz2[i][a,c] +=  KW[i][d,a]*(adjoint(S[i])[e,d])*(adjoint(S[i])[c,d]) * bareta[i][e,d]
        end
    end







    Qz=Qz1-Qz2


    for i in 1:length(QW)
        result+=Qz[i]*conj(KH[i])/m

    end


        #--------------------------------------------------------------------------------------------------------
    # K의 delta를 계산하는 부분


    #첫번째항

    # Kz 계산 (정확한 버전)
    Kz1 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, c in 1:4 , m in 1:4
            Kz1[i][a,c] +=KW[i][m,a]* S[i][c,m]*eta[i][m,c]
        end
    end

    Kz2 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, d in 1:4, c in 1:4 , m in 1:4
            Kz2[i][a,c] +=KW[i][d,a]*S[i][c,m]*S[i][c,d]*eta[i][m,c]
        end
    end
    Kz = Kz1 - Kz2

    # QH 계산
    QH = similar(Kx)

    QH = t


    # # 최종 delta 계산
    for i in 1:length(KW)
        result += Kz[i]*conj(KH[i])/m
    end







    #---------------------------------------------------------------------------------------------------------------------




    Vz=similar(VW)

    for i in 1:lW
        for a in 1:4
            for c in 1:4
                for e in 1:4
                    Vz[i][a,c]+=VW[i][e,a]*S[i][c,e]
                end
            end
        end
    end




    for i in 1:lW
        result+=Vz[i]*conj(dH[i])
    end    




    if layer.residual == true
        result=result+d
        # add!(d,1,result)
    end


    return result

end


function gradient(layer::FullPTCAttention_Layer, ::Softmax, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})




    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W



    lW=length(QW)



    m=layer.nominator


    Hx=layer.V.hop(z)



    Kx=KW*layer.K.hop(z)
    Qx=QW*layer.Q.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 
    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]),Kx[i])/m
        S[i]=layer.activation(M[i])
    end 

    PMK = Vector{Spin_matrix}(undef, lW)

    for i in 1:lW

        PMK[i] =Matmul!(conj(Qx[i]), Hx[i])/m

    end
    
    PMQ = Vector{Spin_matrix}(undef, lW)

    for i in 1:lW


        PMQ[i] =Matmul!(conj(Hx[i]), Kx[i])/m


    end
    





    
    #-------------------------------------------------------------------------------------------------------
    # Q의 graident를 계산하는 부분 


    grad=similar(QW)
    grad1=similar(QW)
    
    for i in 1:lW
        for a in 1:4
            for b in 1:4

                for n in 1:4
                    grad[i][b,a]+=PMQ[i][a,n]*S[i][b,n]*eta[i][n,b]
            
                end

            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for n in 1:4
                    for p in 1:4

                        grad1[i][b,a]+=PMQ[i][a,n]*S[i][b,p]*S[i][b,n]*eta[i][p,b]

                    end
                end
            end
        end
    end 

    gradQ=grad-grad1
    gradQ=conj(gradQ)


#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(QW)
    grad1=similar(QW)




    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for o in 1:4

                    grad[i][a,b]+=PMK[i][o,b]*S[i][o,a]*eta[i][a,o]

                end 
            end
        end
    end 


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for o in 1:4
                    for p in 1:4

                        grad1[i][a,b]+=PMK[i][o,b]*S[i][o,p]*S[i][o,a]*eta[i][p,o]

                    end
                end
            end
        end
    end 


    gradK=grad-grad1


    # gradK=gradK+gradQ




#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    gradV=similar(QW)




    s = layer.V.hop(z)


    for i in 1:lW
        f=Matmul!(Hx[i],d)
        gradV[i]=f*S[i]
    end 

    @inbounds for i in 1:lW
        for u in 1:4
            for v in u+1:4
                tmp = gradV[i][u, v]
                gradV[i][u, v] = gradV[i][v, u]
                gradV[i][v, u] = tmp
            end
        end
    end






    # gradV=similar(gradQ)

    gradlist=[gradQ,gradK,gradV]




    return gradlist


end 

function gradient(layer::FullPTCAttention_Layer, ::Identity, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})
    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W



    lW=length(QW)



    m=layer.nominator


    Hx=layer.V.hop(z)



    Kx=KW*layer.K.hop(z)
    Qx=QW*layer.Q.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 
    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]),Kx[i])/m
        S[i]=layer.activation(M[i])
    end 

    PMK = Vector{Spin_matrix}(undef, lW)

    for i in 1:lW

        PMK[i] =Matmul!(conj(Qx[i]), Hx[i])/m

    end
    
    PMQ = Vector{Spin_matrix}(undef, lW)

    for i in 1:lW


        PMQ[i] =Matmul!(conj(Hx[i]), Kx[i])/m


    end
    





    
    #-------------------------------------------------------------------------------------------------------
    # Q의 graident를 계산하는 부분 


    grad=similar(QW)
    grad1=similar(QW)
    
    for i in 1:lW
        for a in 1:4
            for b in 1:4

                for n in 1:4
                    grad[i][b,a]+=PMQ[i][a,n]*eta[i][n,b]
            
                end

            end
        end
    end 



    gradQ=grad
    gradQ=conj(gradQ)


#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(QW)
    grad1=similar(QW)




    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for o in 1:4

                    grad[i][a,b]+=PMK[i][o,b]*eta[i][a,o]

                end 
            end
        end
    end 


    gradK=grad


    # gradK=gradK+gradQ




#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    gradV=similar(QW)




    s = layer.V.hop(z)


    for i in 1:lW
        f=Matmul!(Hx[i],d)
        gradV[i]=f*S[i]
    end 

    @inbounds for i in 1:lW
        for u in 1:4
            for v in u+1:4
                tmp = gradV[i][u, v]
                gradV[i][u, v] = gradV[i][v, u]
                gradV[i][v, u] = tmp
            end
        end
    end






    gradV=similar(gradQ)
    # gradQ=similar(gradK)
    # gradK=similar(gradK)

    gradlist=[gradQ,gradK,gradV]




    return gradlist
end








function DeltaRule(layer::FullPTCAttention_Layer, ::Softmax, z::WF3x1, d::WF3x1)



    QW = deepcopy(layer.Q.W)
    KW = deepcopy(layer.K.W)
    VW = deepcopy(layer.V.W)

    QW=Clifford(QW)
    KW=Clifford(KW)


    lW=length(QW)

    m=layer.nominator


    Hx=layer.V.hop(z)



    Qx=QW*layer.Q.hop(z)
    Kx=KW*layer.K.hop(z)
    Vx=VW*layer.V.hop(z)


    eta = deepcopy(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 

    bareta = deepcopy(VW)
    for i in 1:lW
        bareta[i]=Matmul!(conj(Vx[i]),conj(d))
    end

    
    # println(eta[1])
    # println(bareta[1])


    
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        M[i]=Matmul!(conj(Qx[i]),Kx[i])/m
        S[i]=layer.activation(M[i])
    end 

  



    hop = layer.Q.hop
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    QH=similar(Hx)
    KH=similar(Hx)
    for i in 1:lW
        KH[i]=invhop(Kx[i])[i]
        # QH[i]=QW[i]*z
        # KH[i]=KW[i]*z
    end
    dH=invhop(conj(d))

    result=similar(d)


    #--------------------------------------------------------------------------------------------------------




    Qz1 = similar(QW)
    Qz2 = similar(QW)


    for i in 1:length(QW)
        for a in 1:4,  c in 1:4, d in 1:4
            Qz1[i][a,c] += KW[i][d,a]*(adjoint(S[i])[c,d])*bareta[i][c,d]
        end
    end

    for i in 1:length(QW)
        for a in 1:4, c in 1:4, e in 1:4 , d in 1:4
            Qz2[i][a,c] +=  KW[i][d,a]*(adjoint(S[i])[e,d])*(adjoint(S[i])[c,d]) * bareta[i][e,d]
        end
    end







    Qz=Qz1-Qz2

    # println(bareta[1])

    for i in 1:length(QW)
        result+=Qz[i]*conj(KH[i])/m

    end


        #--------------------------------------------------------------------------------------------------------
    # K의 delta를 계산하는 부분


    #첫번째항

    # Kz 계산 (정확한 버전)
    Kz1 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, c in 1:4 , m in 1:4
            Kz1[i][a,c] +=KW[i][m,a]* S[i][c,m]*eta[i][m,c]
        end
    end

    Kz2 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, d in 1:4, c in 1:4 , m in 1:4
            Kz2[i][a,c] +=KW[i][d,a]*S[i][c,m]*S[i][c,d]*eta[i][m,c]
        end
    end
    Kz = Kz1 - Kz2

    # QH 계산
    QH = similar(Kx)




    # # 최종 delta 계산
    for i in 1:length(KW)
        result += Kz[i]*conj(KH[i])/m
    end







    #---------------------------------------------------------------------------------------------------------------------




    Vz=similar(VW)

    for i in 1:lW
        for a in 1:4
            for c in 1:4
                for e in 1:4
                    Vz[i][a,c]+=VW[i][e,a]*S[i][c,e]
                end
            end
        end
    end




    for i in 1:lW
        result+=Vz[i]*conj(dH[i])
    end    




    if layer.residual == true
        result=result+d
        # add!(d,1,result)
    end


    return result

end


function DeltaRule(layer::FullPTCAttention_Layer, ::Identity, z::WF3x1, d::WF3x1)

    return nothing
end 





function gradient(layer::PrallelTransportAttention_Layer, ::Identity, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})


    KW = layer.K.W
    VW = layer.V.W



    lW=length(KW)



    m=layer.nominator


    Hx=layer.V.hop(z)



    Kx=KW*layer.K.hop(z)

    Vx=VW*layer.V.hop(z)



    eta=similar(KW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 
    
    M=similar(KW)
    S=similar(KW)
    R=similar(KW)
    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m

    end 

    for i in 1:lW
        for a in 1:4
            S[i][a,a]=M[i][a,a]
        end
    end




    PMK = Vector{Spin_matrix}(undef, lW)


    
    PMQ = Vector{Spin_matrix}(undef, lW)

    for i in 1:lW


        PMK[i] =Matmul!(conj(Hx[1]), Hx[i])/m


    end
    


#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(KW)



    for i in 1:lW
        for a in 1:4
            for b in 1:4
                    grad[i][a,b] = PMK[i][a,b] * eta[i][a,a]
            end
        end
    end


    gradK=grad


    # gradK=gradK+gradQ




#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    gradV=similar(KW)




    s = layer.V.hop(z)

    f=similar(KW)

    for i in 1:lW
        f=Matmul!(Hx[i],d)
        # gradV[i]=f*S[i]
    end 

    # @inbounds for i in 1:lW
    #     for u in 1:4
    #         for v in u+1:4
    #             tmp = gradV[i][u, v]
    #             gradV[i][u, v] = gradV[i][v, u]
    #             gradV[i][v, u] = tmp
    #         end
    #     end
    # end

    for i in 1:lW
        for a in 1:4
            if S[i][a,a] != 0
                for b in 1:4
                        gradV[i][a,b]=f[b,a]*S[i][a,a]
                end
            else
                for b in 1:4
                        gradV[i][a,b]=0
                end
            end
        end
    end


    gradlist=[gradK,gradV]
    return gradlist

end


function DeltaRule(layer::PrallelTransportAttention_Layer, ::Identity, z::WF3x1, d::WF3x1)




    lW=length(layer.V.W)

    m=layer.nominator


    Hx=layer.V.hop(z)


    KW = layer.K.W
    VW = layer.V.W


    Kx=KW*layer.K.hop(z)
    Vx=VW*layer.V.hop(z)


    eta =similar(VW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 

    bareta = deepcopy(VW)
    for i in 1:lW
        bareta[i]=Matmul!(conj(Vx[i]),conj(d))
    end

    
    # println(eta[1])
    # println(bareta[1])


    M=similar(KW)
    S=similar(KW)
    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m
    end 

    for i in 1:lW
        for a in 1:4
            S[i][a,a]=M[i][a,a]
        end
    end



    hop = layer.Q.hop
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    QH=similar(Hx)
    KH=similar(Hx)
    for i in 1:lW
        KH[i]=invhop(Kx[i])[i]
        # QH[i]=QW[i]*z
        # KH[i]=KW[i]*z
    end
    dH=similar(Hx)
    for i in 1:lW
        dH[i]=invhop(z)[i]
    end

    result=similar(d)


    #--------------------------------------------------------------------------------------------------------




    Qz = similar(KW)



    for i in 1:length(KW)
        for a in 1:4  
            Qz[i][a,a] = bareta[i][a,a]
        end
    end









    # println(bareta[1])

    for i in 1:length(KW)
        result+=Qz[i]*conj(Kx[i])/m

    end


        #--------------------------------------------------------------------------------------------------------
    # K의 delta를 계산하는 부분


    #첫번째항

    # Kz 계산 (정확한 버전)
    Kz = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, c in 1:4 
            Kz[i][a,c] =KW[i][c,a]*eta[i][c,c]
        end
    end


    # # 최종 delta 계산
    for i in 1:length(KW)
        result += Kz[i]*conj(dH[i])/m
    end







    #---------------------------------------------------------------------------------------------------------------------




    Vz=similar(VW)

    for i in 1:lW
        for a in 1:4
            for t in 1:4
                    Vz[i][a,t]=VW[i][t,a]*S[i][t,t]
            end
        end
    end




    for i in 1:lW
        result+=Vz[i]*conj(invhop(conj(d))[i])
    end    




    if layer.residual == true
        result=result+d
        # add!(d,1,result)
    end


    return result

end



function gradient(layer::PrallelTransportAttention_Layer, ::ModRelu, z::WF3x1, d::WF3x1, γ::Vector{Spin_matrix})
    
    
    KW = layer.K.W
    VW = layer.V.W

    bias=layer.bias



    lW=length(KW)



    m=layer.nominator


    Hx=layer.V.hop(z)



    Kx=KW*layer.K.hop(z)

    Vx=VW*layer.V.hop(z)

    KH=layer.K.hop(z)

    eta=similar(KW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 


    M=similar(KW)
    S=similar(KW)
    R=similar(KW)
    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m
        for a in 1:4
            R[i][a,a]=norm(M[i][a,a])
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
            val = max( real(R[i][a,a] / ratio[a] - bias), 0)
            S[i][a,a]=val*M[i][a,a]
        end
    end

    # W=S*VW

    # x=similar(z)


    # temp= similar(x)

    # for i in 1:length(KW)
    #     mul!(temp,W[i],Hx[i])
    #     axpby!(1,temp,1,x)
    # end

    # println(norm2(x))



    PMK = Vector{Spin_matrix}(undef, lW)


    


    for i in 1:lW


        PMK[i] =Matmul!(conj(Hx[1]), KH[i])/m


    end


    PMQ= Vector{Spin_matrix}(undef, lW)
    for i in 1:lW
        PMQ[i] =Matmul!(conj(Hx[i]), Hx[1])/m
    end

    

    RM=similar(KW)

    for i in 1:lW
        for a in 1:4
            if real(R[i][a,a]/ratio[a])  > bias
                RM[i][a,a]= (3/2)*R[i][a,a]/ratio[a] -(1/2)*(R[i][a,a]/ratio[a])^2-bias
            else
                RM[i][a,a]=0
            
            end 
        end
    end 








    PM = zeros(Number, lW, lW, 4)

    for k in 1:lW
        for p in 1:lW
            for a in 1:4
                if real(R[p][a,a]/ratio[a])  > bias && k!=p
                    PM[k,p,a] = -abs(R[p][a,a])*M[p][a,a]*conj(M[k][a,a])/(2*(ratio[a])^2*R[k][a,a])
                
                else
                    PM[k,p,a] = 0
                end
            end
        end
    end

    barPM = zeros(Number, lW, lW, 4)

    for k in 1:lW
        for p in 1:lW
            for a in 1:4
                if real(R[p][a,a]/ratio[a])  > bias  && k!=p
                    barPM[k,p,a] = -abs(R[p][a,a])*conj(M[p][a,a])*conj(M[k][a,a])/(2*(ratio[a])^2*R[k][a,a])
                else
                    barPM[k,p,a] = 0
                end
            end
        end
    end


    barRM =similar(KW)


    for i in 1:lW
        for a in 1:4
            if real(R[i][a,a]/ratio[a])  > bias
                barRM[i][a,a]= conj(M[i][a,a])^2/(2*ratio[a]*R[i][a,a])-conj(M[i][a,a])^2/(2*(ratio[a])^2)
            else
                barRM[i][a,a]=0
            
            end 
        end
    end

#---------------------------------------------------------------------------------------------------------------------

    #K의 graident를 계산하는 부분

    grad=similar(KW)
    grad1=similar(KW)
    grad2=similar(KW)
    grad3=similar(KW)

    for i in 1:lW
        for a in 1:4
            for b in 1:4
                    grad[i][a,b] += PMK[i][a,b] *RM[i][a,a]*eta[i][a,a]
            end
        end
    end


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for p in 1:lW
                        grad1[i][a,b] += PM[i,p,a]*PMK[i][a,b]*eta[p][a,a]
                end
            end
        end
    end

    for i in 1:lW
        for a in 1:4
            for b in 1:4
                for p in 1:lW
                        grad2[i][a,b] += barPM[i,p,a]*PMK[i][a,b]*conj(eta[p][a,a])
                end
            end
        end
    end


    for i in 1:lW
        for a in 1:4
            for b in 1:4
                    grad3[i][a,b] += PMK[i][a,b] *barRM[i][a,a]*conj(eta[i][a,a])
            end
        end
    end


    gradK=grad+grad1+grad2+grad3


    # gradK=gradK+gradQ




#----------------------------------------------------------------------------------------------------------------------

    # V의 graident를 계산하는 부분
    

    gradV=similar(KW)




    s = layer.V.hop(z)


    for i in 1:lW
        f=Matmul!(Hx[i],d)
        gradV[i]=f*S[i]
    end 

    @inbounds for i in 1:lW
        for u in 1:4
            for v in u+1:4
                tmp = gradV[i][u, v]
                gradV[i][u, v] = gradV[i][v, u]
                gradV[i][v, u] = tmp
            end
        end
    end


    gradlist=[gradK,gradV]
    return gradlist
    # return result

    
end



function DeltaRule(layer::PrallelTransportAttention_Layer, ::ModRelu, z::WF3x1, d::WF3x1)
    KW = layer.K.W
    VW = layer.V.W

    bias=layer.bias



    lW=length(KW)



    m=layer.nominator


    Hx=layer.V.hop(z)



    Kx=KW*layer.K.hop(z)

    Vx=VW*layer.V.hop(z)

    KH=layer.K.hop(z)

    eta=similar(KW)
    for i in 1:lW
        eta[i]=Matmul!(Vx[i],d)
    end 


    M=similar(KW)
    S=similar(KW)
    R=similar(KW)
    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m
        for a in 1:4
            R[i][a,a]=norm(M[i][a,a])
        end

    end 

    ratio=Vector{Float64}(undef, 4)


    for u in 1:lW
        for a in 1:4
            ratio[a]+=R[u][a,a]
        end
    end

    for i in 1:lW
        for a in 1:4
            val = max( real(R[i][a,a] / ratio[a] - bias), 0)
            S[i][a,a]=val*M[i][a,a]
        end
    end
  



    hop = layer.Q.hop
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    QH=similar(Hx)
    KH=similar(Hx)
    for i in 1:lW
        KH[i]=invhop(Kx[i])[i]
        # QH[i]=QW[i]*z
        # KH[i]=KW[i]*z
    end
    dH=invhop(conj(d))

    result=similar(d)


    #--------------------------------------------------------------------------------------------------------




    Qz1 = similar(QW)
    Qz2 = similar(QW)


    for i in 1:length(QW)
        for a in 1:4,  c in 1:4, d in 1:4
            Qz1[i][a,c] += KW[i][d,a]*(adjoint(S[i])[c,d])*bareta[i][c,d]
        end
    end

    for i in 1:length(QW)
        for a in 1:4, c in 1:4, e in 1:4 , d in 1:4
            Qz2[i][a,c] +=  KW[i][d,a]*(adjoint(S[i])[e,d])*(adjoint(S[i])[c,d]) * bareta[i][e,d]
        end
    end







    Qz=Qz1-Qz2

    # println(bareta[1])

    for i in 1:length(QW)
        result+=Qz[i]*conj(KH[i])/m

    end


        #--------------------------------------------------------------------------------------------------------
    # K의 delta를 계산하는 부분


    #첫번째항

    # Kz 계산 (정확한 버전)
    Kz1 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, c in 1:4 , m in 1:4
            Kz1[i][a,c] +=KW[i][m,a]* S[i][c,m]*eta[i][m,c]
        end
    end

    Kz2 = similar(KW)

    for i in 1:length(KW)
        for a in 1:4, d in 1:4, c in 1:4 , m in 1:4
            Kz2[i][a,c] +=KW[i][d,a]*S[i][c,m]*S[i][c,d]*eta[i][m,c]
        end
    end
    Kz = Kz1 - Kz2

    # QH 계산
    QH = similar(Kx)




    # # 최종 delta 계산
    for i in 1:length(KW)
        result += Kz[i]*conj(KH[i])/m
    end







    #---------------------------------------------------------------------------------------------------------------------




    Vz=similar(VW)

    for i in 1:lW
        for a in 1:4
            for c in 1:4
                for e in 1:4
                    Vz[i][a,c]+=VW[i][e,a]*S[i][c,e]
                end
            end
        end
    end




    for i in 1:lW
        result+=Vz[i]*conj(dH[i])
    end    




    if layer.residual == true
        result=result+d
        # add!(d,1,result)
    end


    return result

end


function gradient(layer::siteweight_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )


    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W


    NX, NY, NZ, NT = z.NX, z.NY, z.NZ, z.NT
    NC, NG = z.NC, z.NG


    lW=length(KW)

    Sbuf = Vector{ComplexF64}(undef, lW) 
    invn = Vector{ComplexF64}(undef, lW)



    # q별 누적 버퍼: v_acc[q] = Σ_p 〈정규화된 S_{p,q}〉 * Hx[q]
    v_acc = [similar(z) for _ in 1:lW]



    m=layer.nominator


    Hx=layer.V.hop(z)

    Qx=QW*layer.Q.hop(z)

    Kx=KW*layer.K.hop(z)

    Vx=VW*layer.V.hop(z)

    KH=layer.K.hop(z)

    eta=Vector{Array{ComplexF64, 4}}(undef, lW)
    for i in 1:lW
        eta[i]=spincolortrace(Vx[i],d)
    end 

    bareta=Vector{Array{ComplexF64, 4}}(undef, lW)
    for i in 1:lW
        bareta[i]=spincolortrace(conj(Vx[i]),conj(d))
    end



# #-------------------------------------------------------------------------------------------------------
    # Q의 graident를 계산하는 부분


    gradQ=similar(KW)
    for u in 1:lW
        for p in 1:lW
            gradQ[u] += Matmul!(conj(Kx[p]),bareta[p]*Hx[u])/m
        end
    end



# 입력:
# Qx, Kx, Vx, Hx :: Vector{WF3x1} (길이 lW)
# eta[q], bareta[q] :: Array{ComplexF64,4}  (spincolortrace로 만든 사이트별 스칼라)
# activation', m, eps
# 출력:
# gradK[q], gradQ[p] :: Spin_matrix (네 타입)





#-------------------------------------------------------------------------------------------------------
    # K의 graident를 계산하는 부분
    gradK=similar(KW)


    for u in 1:lW
        for p in 1:lW
            gradK[u] += Matmul!(conj(Qx[p]),eta[u]*Hx[u])/m
        end
    end






#-------------------------------------------------------------------------------------------------------
    #V의 graident를 계산하는 부분
    gradV=similar(KW)


    @inbounds for k in 1:lW
        v=similar(z)
        a=zeros(ComplexF64,4,4)
        for p in 1:lW
            v += spincolortrace(conj(Qx[p]), Kx[k])*Hx[k]/m
        end
        gradV[k]=Matmul!(d,v)
    end





    # # 3) V의 경로별 gradient
    # for q in 1:lW
    #     gradV[q] = Matmul!(d, v_acc[q])
    # end









    # gradV=similar(KW)
    # gradQ=similar(KW)
    # gradK=similar(KW)


    return [gradQ,gradK,gradV]
end

function gradient(layer::RayleighQuotient_Layer, ::Identity, Dirac::Dirac_operator)


    x=layer.x
    n=norm2(x)


    
    y=Dirac*x
    u=Dirac'*y
    grad = similar(x)
    grad= conj(u)/n- norm2(y)/n^2 * conj(x)




    return [grad]
end

# function gradient(layer::L2Norm_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

#     return nothing
# end 



# function DeltaRule(layer::L2Norm_Layer, ::Identity, z::Vector{WF3x1},batch::Int)

#     @assert length(z) == 1 "L2Norm_Layer DeltaRule expects a single WF3x1 input."

#     result=2*(z[1]-layer.target)/batch

#     return [conj(result)]

# end 






function gradient(layer::PtcRayleighQuotient_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

    return nothing
end 


function DeltaRule(layer::PtcRayleighQuotient_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    result = Vector{WF3x1}(undef, length(z))


    for i in 1:length(z)

        n=norm2(z[i])

        Dirac=layer.D
        y=Dirac*z[i]
        u=Dirac'*y
        result[i] = similar(z[i])
        result[i]=2*(conj(u)/n- norm2(y)/n^2 * conj(z[i]))
    end 




    return result
end

function gradient(layer:: Ritz_Layer , ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

end 




function DeltaRule(layer::Ritz_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    l = length(z)
    D = layer.D


    A = Matrix{ComplexF64}(undef, l, l)
    B = Matrix{ComplexF64}(undef, l, l)
    T = typeof(z[1])
    Dz = Vector{T}(undef, l)
    # Dz 버퍼 한 번만 할당 (similar 대신 deepcopy)
    @inbounds for i in 1:l
        Dz[i] = deepcopy(z[1])
        mul!(Dz[i], D, z[i])   # mul! overwrites Dz[i]
    end
    @inbounds for i in 1:l
        # diagonal
        bii = inner_product(z[i],  z[i])
        aii = inner_product(Dz[i], Dz[i])
        B[i,i] = bii
        A[i,i] = aii

        # upper triangle만 계산하고 Hermitian으로 채우기
        for j in i+1:l
            bij = inner_product(z[i],  z[j])
            aij = inner_product(Dz[i], Dz[j])

            B[i,j] = bij
            B[j,i] = conj(bij)

            A[i,j] = aij
            A[j,i] = conj(aij)
        end
    end
    
    evals, evecs = eigen(A, B)
    @assert length(evals) == l "Eigenvalue decomposition failed in Ritz_Layer."


    m = Vector{ComplexF64}(undef, l)
    for k in 1:l
        m[k] = dot(evecs[:,k], B*evecs[:,k])  # u_k^† B u_k
    end

    # result 버퍼 한 번만 할당
    result = Vector{WF3x1}(undef, l)
    for i in 1:l
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end
    
    # VD 버퍼도 루프 밖에서 미리 할당
    VD = Vector{WF3x1}(undef, l)
    for i in 1:l
        VD[i] = deepcopy(z[1])
    end
    
    # 재사용 버퍼들 - 각 한 번만 할당
    y = deepcopy(z[1])
    f = deepcopy(z[1])
    ψ = deepcopy(z[1])
    r = deepcopy(z[1])
    
    # 모드 k마다 ψ_k 만들고, r_k = D†D ψ_k - λ_k ψ_k 만든다
    for k in 1:l
        zero_wf!(ψ)
        for j in 1:l
            axpby!(evecs[j,k],z[j], 1, ψ)   # ψ = Σ u_{jk} x_j
        end

        mul!(y,D,ψ)                    # y = D ψ_k
        mul!(f,D',y)                   # f = D† D ψ_k
        # r = ψ로 초기화 (inplace copy)
        conj!(r, ψ)
        conj!(r, r)                    # r = ψ (double conj = identity)
        axpby!(1,f,-evals[k],r)        # r = f - λ_k ψ = D†D ψ_k - λ_k ψ_k
        # 네가 "왼쪽작용이 없어 오른쪽으로 하고 다시 conj"라 했으니 그 스타일 유지
        conj!(VD[k], r)
    end

    # ∂/∂V 를 반환하려면 계수는 evecs[i,k] (conj 없음)
    for i in 1:l
        for k in 1:l
            axpby!(evecs[i,k]/m[k],VD[k], 1, result[i])
        end
    end

    return 2*result
end



function DeltaRule!(layer::Ritz_Layer, ::Identity, z::Vector{WF3x1}, delta::Vector{WF3x1}, batch::Int)

    l = length(z)
    D = layer.D
    for i in 1:length(delta)
        zero_wf!(delta[i])  
    end


    A = Matrix{ComplexF64}(undef, l, l)
    B = Matrix{ComplexF64}(undef, l, l)
    T = typeof(z[1])
    Dz = Vector{T}(undef, l)
    # Dz 버퍼 한 번만 할당
    @inbounds for i in 1:l
        Dz[i] = deepcopy(z[1])
        mul!(Dz[i], D, z[i])   # mul! overwrites Dz[i]
    end

    @inbounds for i in 1:l
        # diagonal
        bii = inner_product(z[i],  z[i])
        aii = inner_product(Dz[i], Dz[i])
        B[i,i] = bii
        A[i,i] = aii

        # upper triangle만 계산하고 Hermitian으로 채우기
        for j in i+1:l
            bij = inner_product(z[i],  z[j])
            aij = inner_product(Dz[i], Dz[j])

            B[i,j] = bij
            B[j,i] = conj(bij)

            A[i,j] = aij
            A[j,i] = conj(aij)
        end
    end
    
    evals, evecs = eigen(A, B)
    @assert length(evals) == l "Eigenvalue decomposition failed in Ritz_Layer."


    m = Vector{ComplexF64}(undef, l)
    for k in 1:l
        m[k] = dot(evecs[:,k], B*evecs[:,k])  # u_k^† B u_k
    end

    # VD 버퍼 루프 밖에서 미리 할당
    VD = Vector{WF3x1}(undef, l)
    for i in 1:l
        VD[i] = deepcopy(z[1])
    end

    # 재사용 버퍼들 - 각 한 번만 할당
    y = deepcopy(z[1])
    f = deepcopy(z[1])
    ψ = deepcopy(z[1])
    r = deepcopy(z[1])
    
    # 모드 k마다 ψ_k 만들고, r_k = D†D ψ_k - λ_k ψ_k 만든다
    for k in 1:l
        zero_wf!(ψ)
        for j in 1:l
            axpby!(evecs[j,k],z[j], 1, ψ)   # ψ = Σ u_{jk} x_j
        end

        mul!(y,D,ψ)                    # y = D ψ_k
        mul!(f,D',y)                   # f = D† D ψ_k
        # r = ψ로 초기화 (inplace copy)
        conj!(r, ψ)
        conj!(r, r)                    # r = ψ (double conj = identity)
        axpby!(1,f,-evals[k],r)        # r = f - λ_k ψ = D†D ψ_k - λ_k ψ_k
        # 네가 "왼쪽작용이 없어 오른쪽으로 하고 다시 conj"라 했으니 그 스타일 유지
        conj!(VD[k], r)
    end

    # ∂/∂V 를 반환하려면 계수는 evecs[i,k] (conj 없음)
    for i in 1:l
        for k in 1:l
            axpby!(2*evecs[i,k]/m[k], VD[k], 1, delta[i])
        end
    end

    # for i in 1:l
    #     axpby!(0,delta[i], 1/batch, delta[i])  # batch로 나누기
    # end



end






function gradient(layer:: Orthogonal_Layer , ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})


end

function DeltaRule(layer::Orthogonal_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    result = Vector{WF3x1}(undef, length(z))
    temp = similar(z[1])
    for i in 1:length(z)
        result[i]=deepcopy(temp)
    end

    a=layer.coeff

    l=length(z)
    cl=(l-1)*l/2


    
    for i in 2:l
        for j in 1:i-1
            c=inner_product(z[j],z[i])
            result[i]+=a*c'*conj(z[j])
            result[j]+=a*c*conj(z[i])
        end
    end 

    return 2*result
end



function gradient(layer::HoppingRayleighQuotient_Layer, ::Identity, Dirac::Dirac_operator)


    z=layer.hop(layer.x)


    x=similar(layer.x)
    y=similar(layer.x)

    
    for i in 1:length(z)
        axpby!(1,z[i],1,x)
    end 
    y=Dirac*x

    u=Dirac'*x
    invP = reverse_paths(layer.hop.P)
    invhop = Hopping_operator(layer.hop.Gauge, invP)
    phi=invhop(u)
    c=invhop(x)




    result=similar(y)
    for i in 1:length(z)
        result+=conj(phi[i])/norm2(x)-(norm2(y)/norm2(x)^2)*conj(c[i])
    end

    return [result]
end


# function gradient(layer::DenseLinearLayer, ::Identity, z::Vector{WF3x1CPU}, d::Vector{T} ,  γ::Vector{Spin_matrix} ) where T<:WF3x1


#     grad=Vector{Spin_matrix}(undef, length(layer.W))

#     for i in 1:length(layer.W)
#         grad[i]=similar(layer.W[i])
#     end



#     for i in 1:length(layer.W)
#         grad[i]=Matmul!(d[i],z[i])
#     end

#     return [grad]

function gradient(layer::DenseLinearLayer, ::Identity,
                  z::AbstractVector{TZ},
                  d::AbstractVector{TD},
                  M::Any) where {TZ<:WF3x1, TD<:WF3x1}
    
    grad=Vector{typeof(layer.W[1])}(undef, length(layer.W))

    for i in 1:length(layer.W)
        grad[i]=similar(layer.W[i])
    end

    for k in 1:length(z)

        for i in 1:length(d)
            grad[i+(k-1)*length(d)]=Matmul!(d[i],z[k])
        end
    end 


    return grad
end

function DeltaRule(layer::DenseLinearLayer, ::Identity, z::Vector{T}, d::Vector{WF3x1} ) where T<:WF3x1

    W=layer.W


    result=Vector{WF3x1}(undef,length(z))
    y=similar(z[1])
    for i in 1:length(z)
        result[i]=deepcopy(y)
    end

    for k in 1:length(z)

        for i in 1:length(d)

            # addmul!(result,adjoint(W[i]),(conj(d[i])))

            mul!(y,adjoint(W[i+(k-1)*length(d)]),(conj(d[i])))

            axpby!(1,conj(y),1,result[k])
            # result=result+y
        end 
    end 
    return result
end

function DeltaRule!(layer::DenseLinearLayer, ::Identity, z::Vector{T}, d::Vector{WF3x1}, delta::Vector{WF3x1}) where T<:WF3x1

    W=layer.W
    # result=Vector{WF3x1}(undef,length(z))
    y=similar(z[1])
    # for i in 1:length(z)
    #     delta[i]=deepcopy(y)
    # end
    zero_wf!(delta)
    for k in 1:length(z)

        for i in 1:length(d)

            # addmul!(result,adjoint(W[i]),(conj(d[i])))

            mul!(y,adjoint(W[i+(k-1)*length(d)]),(conj(d[i])))

            axpby!(1,conj(y),1,delta[k])
            # result=result+y
        end 
    end 
    return nothing
end






function gradient(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    Out=layer.Out
    W=layer.W
    paths=layer.paths

    hop=Hopping_operator(layer.U,paths[1])

    n=length(layer.W)

    @assert length(z) == length(paths) "PTLayer gradient: z의 길이는 paths 길이와 같아야 합니다. length(z)=$(length(z)), length(paths)=$(length(paths))"
    @assert length(d) == Out "PTLayer gradient: d의 길이는 Out과 같아야 합니다. length(d)=$(length(d)), Out=$Out"
    @assert length(W) == length(paths)*Out "The length of W must be equal to length(paths)*Out."

    # grad 벡터 할당 최소화 (similar 대신 deepcopy)
    grad=Vector{typeof(W[1])}(undef, n)
    for i in 1:n
        grad[i]=deepcopy(W[1])
    end 
    for i in 1:length(paths)
        hop.P=paths[i]
        u = hop(z[i])
        for j in 1:Out
            grad[(i-1)*Out+j]=Matmul!(d[j],(u[1]))
        end
    end 


    return [grad]


end 




function DeltaRule(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)
    # result 버퍼 할당 (similar 제거, deepcopy만 사용)
    result = Vector{WF3x1}(undef, Nz)
    for i in 1:Nz
        result[i] = deepcopy(z[1])
    end

    # ① d_conj[j] = conj(d[j]) 미리
    d_conj = Vector{WF3x1}(undef, Nd)
    for i in 1:Nd
        d_conj[i] = deepcopy(z[1])
    end
    if d[1] isa WF3x1CPU
        @inbounds for j in 1:Nd
            conj!(d_conj[j], d[j])
        end
    else
        @inbounds for j in 1:Nd
            conj!(d_conj[j], d[j])
        end
    end

    # ② 역경로 미리
    invpaths = Vector{typeof(paths[1])}(undef, Nz)
    @inbounds for i in 1:Nz
        invpaths[i] = reverse_paths(paths[i])
    end

    # ③ Wadj[k] = adjoint(W[k]) 미리
    Wadj = Vector{typeof(W[1])}(undef, NW)
    @inbounds for k in 1:NW
        Wadj[k] = adjoint(W[k])
    end

    # ④ 스레드별로 쓸 invhop 준비 (공유 금지)
    invhops = [Hopping_operator(layer.U, invpaths[1]) for _ in 1:nthreads()]

    # ⑤ i 방향으로 병렬화
    if z[1] isa WF3x1CPU
        # 스레드별 버퍼 미리 할당 (similar 대신 각 스레드당 하나씩)
        ri_buffers = [deepcopy(z[1]) for _ in 1:nthreads()]
        
        @threads for i in 1:Nz
            # 이 스레드 전용 hop 사용
            invhop = invhops[threadid()]
            invhop.P = invpaths[i]
            ri = ri_buffers[threadid()]
            zero_wf!(ri)  # 버퍼 재사용을 위해 초기화
            base = (i-1) * Out
            @inbounds for j in 1:Out
            hopped = invhop(d_conj[j])[1]
            addmul!(ri, Wadj[base + j], hopped)
            end

            # 원래 코드와 같은 의미: result[i] = conj(result[i])
            # result[i] = conj(ri)
            conj!(result[i], ri)
        end
    else
        # GPU는 병렬화 안 되므로 하나의 버퍼만 필요
        ri = deepcopy(z[1])
        
        for i in 1:Nz
        # 이 스레드 전용 hop 사용
        invhop = invhops[threadid()]
        invhop.P = invpaths[i]
        zero_wf!(ri)  # 버퍼 재사용을 위해 초기화
        base = (i-1) * Out
        @inbounds for j in 1:Out
            hopped = invhop(d_conj[j])[1]    
            gpuaddmul!(ri,Wadj[base + j],hopped)
            end
            conj!(result[i], ri)
        end

    end


    return result
end



function DeltaRule!(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."
    @assert length(delta) == length(z) "The length of delta must be equal to the length of z."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)
    # temp 제거, 직접 deepcopy 사용

    # ① d_conj[j] = conj(d[j]) 미리
    d_conj = Vector{WF3x1}(undef, Nd)

    @inbounds for i in 1:Nd
        d_conj[i] = deepcopy(z[1])
        conj!(d_conj[i], d[i])
    end


    # ② 역경로 미리
    invpaths = Vector{typeof(paths[1])}(undef, Nz)
    @inbounds for i in 1:Nz
        invpaths[i] = reverse_paths(paths[i])
    end

    # ③ Wadj[k] = adjoint(W[k]) 미리
    Wadj = Vector{typeof(W[1])}(undef, NW)
    @inbounds for k in 1:NW
        Wadj[k] = adjoint(W[k])
    end

    # ④ 스레드별로 쓸 invhop 준비 (공유 금지)
    invhops = [Hopping_operator(layer.U, invpaths[1]) for _ in 1:nthreads()]

    # ⑤ i 방향으로 병렬화
    if z[1] isa WF3x1CPU
        # 스레드별 버퍼 미리 할당
        ri_buffers = [deepcopy(z[1]) for _ in 1:nthreads()]
        
        @threads for i in 1:Nz
            # 이 스레드 전용 hop 사용
            invhop = invhops[threadid()]
            invhop.P = invpaths[i]
            ri = ri_buffers[threadid()]
            zero_wf!(ri)  # 버퍼 재사용을 위해 초기화
            base = (i-1) * Out
            @inbounds for j in 1:Out
            hopped = invhop(d_conj[j])[1]
            addmul!(ri, Wadj[base + j], hopped)
            end

            # 원래 코드와 같은 의미: result[i] = conj(result[i])
            # result[i] = conj(ri)
            conj!(delta[i], ri)
        end
    else
        # GPU는 병렬화 안 되므로 하나의 버퍼만 필요
        ri = deepcopy(z[1])
        
        for i in 1:Nz
        # 이 스레드 전용 hop 사용
        invhop = invhops[threadid()]
        invhop.P = invpaths[i]
        zero_wf!(ri)  # 버퍼 재사용을 위해 초기화
        base = (i-1) * Out
        @inbounds for j in 1:Out
            hopped = invhop(d_conj[j])[1]    
            gpuaddmul!(ri,Wadj[base + j],hopped)
            end
            conj!(delta[i], ri)
        end

    end

    return nothing
end



function gradient(layer::fermionfield_Layer, ::Identity, z::Any, d::Vector{T} ,  γ::Vector{Spin_matrix} ) where T<:WF3x1

    return d
    
end


function DeltaRule(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1)
    
end

function DeltaRule!(layer::fermionfield_Layer, ::Identity, z::Any, d::WF3x1, delta::Any)
    
end


function gradient(L::LossSumLayer, ::ActivationFunction, z, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end




# 출력층 DeltaRule: 각 loss의 델타를 더하면 됨 (∇(w1 f + w2 g) = w1∇f + w2∇g)
function DeltaRule(L::LossSumLayer,  ::Identity, z::Vector{WF3x1}, batch::Int)
    d1 = DeltaRule(L.l1, L.l1.activation, z, batch)
    d2 = DeltaRule(L.l2, L.l2.activation, z, batch)
    @assert length(d1) == length(d2) "DeltaRule outputs from loss layers must have the same length."
    return d1 + d2
end














function gradient(layer::Dirac_Layer,::Any,z::WF3x1,d::WF3x1 ,  γ::Vector{Spin_matrix} )


    return nothing
end 

function DeltaRule(layer::Dirac_Layer, ::Any,  z::WF3x1, d::WF3x1)

    D=layer.activation
    cd=conj(d)
    y=similar(d)
    mul!(y,D',cd)
    return conj(y)
end 



function gradient(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    return nothing
end


function DeltaRule(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1



    result = Vector{WF3x1}(undef, length(z))
    α = layer.alpha


    @inbounds for i in 1:length(z)
        cz=conj(z[i])
        norm = sqrt(norm2(z[i]))
        eta1 = inner_product(cz, d[i])/2                  # z† d
        eta2 = inner_product(z[i], conj(d[i]))/2            # zᵗ d*

        result[i] = d[i] / norm
        result[i] -= (eta1 * cz) / norm^3
        result[i] -= (eta2 * cz) / norm^3
    end

    return result
end

function DeltaRule!(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1



    # result = Vector{WF3x1}(undef, length(z))


    α = layer.alpha


    @inbounds for i in 1:length(z)
        cz=conj(z[i])
        norm = sqrt(norm2(z[i]))
        eta1 = inner_product(cz, d[i])/2                  # z† d
        eta2 = inner_product(z[i], conj(d[i]))/2            # zᵗ d*

        delta[i] = d[i] / norm
        delta[i] -= (eta1 * cz) / norm^3
        delta[i] -= (eta2 * cz) / norm^3
    end


end





function gradient(layer::complexNormLayer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end

function DeltaRule(layer::complexNormLayer, ::Identity,z::WF3x1, d::WF3x1)

    m=mean(d)

    

    result=d-m


    return result

end 



function gradient(layer::zscoreNormLayer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end


function DeltaRule(layer::zscoreNormLayer, ::Identity,z::WF3x1, d::WF3x1)


    real=(z+conj(z))/2
    imag=(z-conj(z))/2im

    mu=mean(z)

    rd=similar(d)
    id=similar(d)
    N=length(z)

    rσ=sqrt(norm2(real-mu.real)/N)
    iσ=sqrt(norm2(imag-mu.imag)/N)


    #---------------------------------------------------------------------------------
    # real 미분


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


function gradient(layer::GramSchmidt_Layer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    return nothing
end 
function DeltaRule(layer::GramSchmidt_Layer, ::Identity,
                   z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1

    V = layer.x
    l = length(d)

    result = Vector{WF3x1}(undef, l)

    @inbounds for i in 1:l
        g = deepcopy(d[i])
        for j in 1:length(V)
            denom = inner_product(V[j], V[j])
            coeff = inner_product(V[j], g) / denom
            axpby!(-coeff, V[j], 1, g)   # g -= coeff * V[j]
        end
        result[i] = g
    end

    return result
end
function DeltaRule!(layer::GramSchmidt_Layer, ::Identity,
                   z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    V = layer.x
    l = length(d)

    @inbounds for i in 1:l
        g = deepcopy(d[i])
        for j in 1:length(V)
            denom = inner_product(V[j], V[j])
            coeff = inner_product(V[j], g) / denom
            axpby!(-coeff, V[j], 1, g)   # g -= coeff * V[j]
        end
        delta[i] = g
    end

end




end # module activation






