module activation


# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using CUDA
using Base.Threads
using StaticArrays
  # 활성화 함수 추상 타입


import ..elementry_base: Spin_matrix, inner_product, Clifford, reverse_paths,Hopping_operator,norm2,mean,minmax,spincolortrace,addmul!,substitute_fermion!
import ..Layer_types: CPTC_Layer, PTC_Layer, Layer,ActivationFunction,Identity ,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer,minmaxNormLayer,zscoreNormLayer,siteweight_Layer
import ..Layer_types: FullPTCAttention_Layer,Dirac_Layer,PrallelTransportAttention_Layer,ModRelu,RayleighQuotient_Layer,PtcRayleighQuotient_Layer
import ..Layer_types: HoppingRayleighQuotient_Layer,DenseLinearLayer,PTLayer,fermionfield_Layer,LossSumLayer,Orthogonal_Layer,Ritz_Layer,GramSchmidt_Layer,BlockScalar_Layer,BlockMatrix_Layer
import ..elementry_base: blockize, unblockize
import ..gpu: Matmul!,Spin_matrix_GPU,gpuaddmul!,zero_wf!,conj!,Matmul_batch,Matmul_batch!,_build_block_weight_gpu,_init_gpu_block_cache!,_update_block_weight_gpu!,_sync_blockscalar_W_gpu!,blockscalar_apply_gpu!,blockscalar_gradient_gpu!,_sync_blockmatrix_W_gpu!,blockmatrix_apply_gpu!,blockmatrix_gradient_gpu!,blockmatrix_delta_gpu!
import ..gpu: pack_fermions!,unpack_fermions!,batched_gradient!,batched_delta!,sync_Wadj_gpu!,fused_hopping_gpu!,sync_W_to_gpu!,batched_forward!,batched_inv_hopping_gpu!,get_shared_buf!,get_shared_float_buf!,batched_hopping_steps_gpu!,_batch_normsq_dot!,_backward_normalize!,_sum_spin_color!,inplace_conj!,_normlayer_bwd!

export gradient, DeltaRule,DeltaRule!
export Identity, ReLU, Sigmoid, Tanh

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

import ..gpu: _alloc_result_wf





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

    # forward에서 캐싱된 결과 재사용
    evals  = layer._evals
    evecs  = layer._evecs
    m      = layer._m
    DdagDz = layer._DdagDz

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
    
    ψ = deepcopy(z[1])
    r = deepcopy(z[1])
    
    # D†D ψ_k = Σ_j u_{jk} D†D z_j (Dirac ops 0회)
    for k in 1:l
        zero_wf!(ψ)
        zero_wf!(r)
        for j in 1:l
            axpby!(evecs[j,k], z[j], 1, ψ)
            axpby!(evecs[j,k], DdagDz[j], 1, r)
        end
        axpby!(1, r, -evals[k], ψ)
        conj!(VD[k], ψ)
    end

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

    # forward에서 캐싱된 eigen 결과 재사용 (Gram matrix + inner_product 6회 + eigen 재계산 제거)
    evals = layer._evals
    evecs = layer._evecs
    m     = layer._m
    DdagDz = layer._DdagDz   # D†D z[j] — forward에서 캐시됨

    # VD, ψ, r 버퍼 캐시 (매 epoch deepcopy 방지)
    need_vd_reinit = isempty(layer._VD) || length(layer._VD) != l
    if !need_vd_reinit
        need_vd_reinit = typeof(layer._VD[1]) != typeof(z[1])
    end
    if need_vd_reinit
        layer._VD = Vector{Any}(undef, l)
        for i in 1:l
            layer._VD[i] = _alloc_result_wf(z[1])
        end
        layer._work = Any[_alloc_result_wf(z[1]), _alloc_result_wf(z[1])]  # [ψ, r] only (y, f 불필요)
    end

    VD = layer._VD
    ψ  = layer._work[1]
    r  = layer._work[2]
    
    # D†D ψ_k = Σ_j u_{jk} D†D z_j (Dirac ops 0회 — DdagDz 캐시 재사용)
    for k in 1:l
        zero_wf!(ψ)
        zero_wf!(r)
        for j in 1:l
            axpby!(evecs[j,k], z[j], 1, ψ)         # ψ = Σ u_{jk} z_j
            axpby!(evecs[j,k], DdagDz[j], 1, r)     # r = Σ u_{jk} D†D z_j = D†D ψ_k
        end
        # r = D†D ψ_k - λ_k ψ_k
        axpby!(1, r, -evals[k], ψ)    # ψ = r - λ_k ψ → but we want r = D†Dψ - λψ
        # 위에서 axpby!(1,r,-λ,ψ) → ψ = 1*r + (-λ)*ψ = D†Dψ - λψ. 맞음
        conj!(VD[k], ψ)
    end

    # ∂/∂V 를 반환하려면 계수는 evecs[i,k] (conj 없음)
    for i in 1:l
        for k in 1:l
            axpby!(2*evecs[i,k]/m[k], VD[k], 1, delta[i])
        end
    end



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
    
    n = length(layer.W)
    nd = length(d)
    nz = length(z)

    if d[1] isa WF3x1GPU && layer._W_gpu !== nothing
        # GPU batched gradient: 1회 커널
        NC = 3; NG = 4
        blocksize = d[1].blockinfo.blocksize
        rsize     = d[1].blockinfo.rsize

        if layer._grad_gpu === nothing
            layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, n)
        end
        # d_buf, input_all should already be packed from forward/backward
        # but repack to be safe (d may differ from backward's d)
        if layer._d_buf === nothing || size(layer._d_buf, 5) != nd
            layer._d_buf = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, nd)
        end
        pack_fermions!(layer._d_buf, d)

        if layer._input_all === nothing || size(layer._input_all, 5) != nz
            layer._input_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, nz)
        end
        pack_fermions!(layer._input_all, z)

        batched_gradient!(layer._grad_gpu, layer._d_buf, layer._input_all,
                          blocksize, rsize, nd, nz)

        return [layer._grad_gpu]
    else
        grad = Vector{typeof(layer.W[1])}(undef, n)
        @threads for idx in 1:n
            i = ((idx-1) % nd) + 1
            k = ((idx-1) ÷ nd) + 1
            grad[idx] = Matmul!(d[i], z[k])
        end
        return grad
    end
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

function DeltaRule!(layer::DenseLinearLayer, ::Identity, z::Vector{T}, d::Vector{<:WF3x1}, delta::Vector{<:WF3x1}) where T<:WF3x1

    W   = layer.W
    In  = layer.In
    Out = layer.Out
    NW  = length(W)

    if z[1] isa WF3x1GPU && layer._W_gpu !== nothing
        # GPU batched: Wadj * conj(d) → delta, 1회 커널
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        if layer._Wadj_gpu === nothing
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)

        if layer._d_buf === nothing || size(layer._d_buf, 5) != Out
            layer._d_buf = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, Out)
        end
        pack_fermions!(layer._d_buf, d)

        if layer._ci_all === nothing || size(layer._ci_all, 5) != In
            layer._ci_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, In)
        end

        batched_delta!(layer._ci_all, layer._Wadj_gpu, layer._d_buf, blocksize, rsize, Out, In)
        unpack_fermions!(delta, layer._ci_all)
        # batched_delta!는 Wadj*conj(d)만 계산 — CPU 경로의 conj(W†*conj(d))와 맞추려면 outer conj 필요
        for k in 1:In
            conj!(delta[k], delta[k])
        end
    else
        # CPU path
        if !layer._delta_init_dl || typeof(layer._conj_buf[1]) != typeof(z[1])
            layer._conj_buf = [deepcopy(z[1]), deepcopy(z[1])]  # [conj_d, conj_y]
            layer._delta_init_dl = true
        end
        conj_d = layer._conj_buf[1]
        conj_y = layer._conj_buf[2]
        y = layer._yi[1]

        zero_wf!(delta)
        for k in 1:In
            for i in 1:Out
                conj!(conj_d, d[i])
                mul!(y, adjoint(W[i+(k-1)*Out]), conj_d)
                conj!(conj_y, y)
                axpby!(1, conj_y, 1, delta[k])
            end
        end
    end
    return nothing
end






function gradient(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1

    Out=layer.Out
    W=layer.W
    paths=layer.paths

    n=length(layer.W)
    npaths = length(paths)

    @assert length(z) == npaths "PTLayer gradient: z의 길이는 paths 길이와 같아야 합니다. length(z)=$(length(z)), length(paths)=$(npaths)"
    @assert length(d) == Out "PTLayer gradient: d의 길이는 Out과 같아야 합니다. length(d)=$(length(d)), Out=$Out"
    @assert length(W) == npaths*Out "The length of W must be equal to length(paths)*Out."

    # gradient checkpointing: hopped_all은 per-layer로 저장하지 않고 z에서 재계산
    # (9 PTLayer × 628MB = 5.6GB 절약 — backward는 sequential이므로 shared buffer 안전)
    if layer._W_gpu !== nothing
        if d[1] isa WF3x1GPU
            # GPU batched: 625개 kernel → 1회 launch, GPU→CPU 전송 없음
            blocksize = d[1].blockinfo.blocksize
            rsize     = d[1].blockinfo.rsize
            NC = 3; NG = 4

            # forward에서 캐싱된 hop 결과 재사용 (hop 재계산 방지)
            if layer._hopped_all !== nothing && size(layer._hopped_all, 5) == npaths
                _hopped_g = layer._hopped_all
            else
                _hopped_g = get_shared_buf!(:hopped_all, NC, NG, blocksize, rsize, npaths)
                _hop_t    = get_shared_buf!(:hop_temp,   NC, NG, blocksize, rsize, npaths)
                pack_fermions!(_hopped_g, z)
                if layer._hop_max_steps > 0 && layer._hop_dirs_gpu !== nothing
                    batched_hopping_steps_gpu!(
                        _hopped_g, _hop_t, z,
                        layer.U, layer._hop_dirs_gpu, layer._hop_nsteps_gpu,
                        layer._hop_max_steps, npaths, blocksize, rsize)
                end
            end

            # d_buf: 전역 공유 버퍼 사용
            _d_buf_g = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Out)
            if layer._grad_gpu === nothing || size(layer._grad_gpu, 3) != n
                layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, n)
            end

            pack_fermions!(_d_buf_g, d)
            batched_gradient!(layer._grad_gpu, _d_buf_g, _hopped_g,
                              blocksize, rsize, Out, npaths)

            return [layer._grad_gpu]
        elseif length(layer._hopped) == npaths
            # CPU: @threads 병렬화 (CPU path는 _hopped 캐시 그대로 사용)
            grad = Vector{typeof(W[1])}(undef, n)
            @threads for idx in 1:n
                j = ((idx-1) % Out) + 1
                i = ((idx-1) ÷ Out) + 1
                grad[idx]=Matmul!(d[j], layer._hopped[i])
            end
            return [grad]
        end
    else
        grad = Vector{typeof(W[1])}(undef, n)
        hop=Hopping_operator(layer.U,paths[1])
        for i in 1:npaths
            hop.P=paths[i]
            u = hop(z[i])
            if d[1] isa WF3x1GPU
                Qs = Vector{WF3x1GPU}(undef, Out)
                Ks = Vector{WF3x1GPU}(undef, Out)
                @inbounds for j in 1:Out
                    Qs[j] = d[j]
                    Ks[j] = u[1]
                end
                batch_result = Matmul_batch(Qs, Ks)
                @inbounds for j in 1:Out
                    grad[(i-1)*Out+j] = batch_result[j]
                end
            else
                @threads for j in 1:Out
                    grad[(i-1)*Out+j]=Matmul!(d[j],(u[1]))
                end
            end
        end
        return [grad]
    end

end 




function DeltaRule(layer::PTLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1

    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    @assert length(d) == Out "The length of d must be equal to Out."

    Nz = length(z)
    Nd = length(d)
    NW = length(W)
    result = Vector{WF3x1}(undef, Nz)
    for i in 1:Nz
        result[i] = deepcopy(z[1])
        zero_wf!(result[i])
    end

    # ① d_conj 버퍼 캐시 — CPU path에서만 필요
    is_gpu = z[1] isa WF3x1GPU && layer._W_gpu !== nothing

    if !is_gpu
        if !layer._delta_init || length(layer._d_conj_buf) != Nd
            layer._d_conj_buf = Vector{WF3x1}(undef, Nd)
            for i in 1:Nd
                layer._d_conj_buf[i] = deepcopy(d[1])
            end
            layer._ci_buffers = [deepcopy(z[1]) for _ in 1:nthreads()]
            layer._delta_init = true
        end
        d_conj = layer._d_conj_buf
        @inbounds for j in 1:Nd
            conj!(d_conj[j], d[j])
        end
    elseif !layer._delta_init
        layer._delta_init = true
    end

    # ② 역경로 캐시 (paths 불변이므로 1회만 계산)
    if isempty(layer._invpaths) || length(layer._invpaths) != Nz
        layer._invpaths = Vector{typeof(paths[1])}(undef, Nz)
        @inbounds for i in 1:Nz
            layer._invpaths[i] = reverse_paths(paths[i])
        end
    end
    invpaths = layer._invpaths

    # ③ Wadj 캐시 — CPU path에서만 필요
    if !is_gpu
        if length(layer._Wadj) != NW
            layer._Wadj = Vector{typeof(W[1])}(undef, NW)
            @inbounds for k in 1:NW
                layer._Wadj[k] = adjoint(W[k])
            end
        end
        Wadj = layer._Wadj
        @inbounds for k in 1:NW
            wsm = W[k].sm
            wadj_sm = Wadj[k].sm
            for j in 1:4
                for i in 1:4
                    wadj_sm[i,j] = conj(wsm[j,i])
                end
            end
        end
    end

    # ④ invhops + hop 출력 버퍼 캐시 (CPU path에서만 필요)
    if !is_gpu
        nt = nthreads()
        if isempty(layer._invhops) || length(layer._invhops) != nt
            layer._invhops = [Hopping_operator(layer.U, invpaths[1]) for _ in 1:nt]
            layer._hop_out = [Vector{WF3x1}([deepcopy(z[1])]) for _ in 1:nt]
        else
            @inbounds for t in 1:nt
                layer._invhops[t].Gauge = layer.U
            end
        end
    end

    # ⑤ 최적화: 먼저 c_i = Σ_j Wadj_{ij} * d_conj[j] 계산 후, hop 1회만 수행
    if z[1] isa WF3x1CPU
        ci_buffers = layer._ci_buffers
        invhops = layer._invhops
        hop_out = layer._hop_out

        @threads for i in 1:Nz
            tid = threadid()
            ci = ci_buffers[tid]
            zero_wf!(ci)
            base = (i-1) * Out
            @inbounds for j in 1:Out
                addmul!(ci, Wadj[base + j], d_conj[j])
            end
            invhop = invhops[tid]
            invhop.P = invpaths[i]
            invhop(ci, hop_out[tid])  # preallocated 버전 사용
            conj!(result[i], hop_out[tid][1])
        end
    else
        # GPU batched: Wadj*conj(d) 를 1회 커널로
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        if layer._W_gpu !== nothing
            sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)
        else
            Wadj_cpu = Array{ComplexF64}(undef, NG, NG, NW)
            @inbounds for k in 1:NW
                sm = layer._Wadj[k].sm
                for jj in 1:4, ii in 1:4
                    Wadj_cpu[ii, jj, k] = sm[ii, jj]
                end
            end
            copyto!(layer._Wadj_gpu, Wadj_cpu)
        end

        # d_buf, ci_all: 공유 임시 버퍼 사용 (layers 간 재사용)
        _d_buf = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Nd)
        pack_fermions!(_d_buf, d)

        _ci_buf = get_shared_buf!(:ci_all, NC, NG, blocksize, rsize, Nz)

        # 1회 batched kernel: ci_all = Σ_j Wadj * conj(d)
        batched_delta!(_ci_buf, layer._Wadj_gpu, _d_buf, blocksize, rsize, Nd, Nz)

        # lazy init inverse direction table (forward init이 안 된 경우 여기서 수행)
        if layer._inv_hop_dirs_gpu === nothing
            max_steps = layer._hop_max_steps
            if max_steps == 0 && layer._hop_dirs_gpu === nothing
                max_steps = 0
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0; ns = 0; end
                    max_steps = max(max_steps, ns)
                end
                layer._hop_max_steps = max_steps

                dirs_cpu = zeros(Int8, max_steps, Nz)
                nsteps_cpu = zeros(Int32, Nz)
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        nsteps_cpu[i] = Int32(0)
                    else
                        nsteps_cpu[i] = Int32(ns)
                        for s in 1:ns; dirs_cpu[s, i] = Int8(steps_i[s].f); end
                    end
                end
                layer._hop_dirs_gpu   = CuArray(dirs_cpu)
                layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
            end

            max_steps = layer._hop_max_steps
            inv_dirs_cpu = zeros(Int8, max_steps, Nz)
            for i in 1:Nz
                inv_steps = invpaths[i].various[1].steps
                ns = length(inv_steps)
                if !(ns == 1 && inv_steps[1].f == 0)
                    for s in 1:ns; inv_dirs_cpu[s, i] = Int8(inv_steps[s].f); end
                end
            end
            layer._inv_hop_dirs_gpu = CuArray(inv_dirs_cpu)
        end

        # batched inverse hop: ci_buf ↔ shared hop_temp
        max_steps = layer._hop_max_steps
        if max_steps > 0
            _hop_temp = get_shared_buf!(:hop_temp, NC, NG, blocksize, rsize, Nz)
            bi = z[1].blockinfo
            NXl = z[1].NX; NYl = z[1].NY; NZl = z[1].NZ; NTl = z[1].NT
            batched_inv_hopping_gpu!(
                _ci_buf, _hop_temp, layer.U,
                layer._inv_hop_dirs_gpu, layer._hop_nsteps_gpu,
                max_steps, Nz, blocksize, rsize, bi, NXl, NYl, NZl, NTl)
        end

        # batch conj
        inplace_conj!(_ci_buf)

        # unpack → result
        unpack_fermions!(result, _ci_buf)
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

    # ① d_conj 버퍼 캐시 — CPU path에서만 필요
    is_gpu = z[1] isa WF3x1GPU && layer._W_gpu !== nothing

    if !is_gpu
        if !layer._delta_init || length(layer._d_conj_buf) != Nd
            layer._d_conj_buf = Vector{WF3x1}(undef, Nd)
            for i in 1:Nd
                layer._d_conj_buf[i] = deepcopy(d[1])
            end
            layer._ci_buffers = [deepcopy(z[1]) for _ in 1:nthreads()]
            layer._delta_init = true
        end
        d_conj = layer._d_conj_buf
        @inbounds for i in 1:Nd
            conj!(d_conj[i], d[i])
        end
    elseif !layer._delta_init
        # GPU path: ci_buffers, invhops, hop_out 불필요 (batched 처리)
        layer._delta_init = true
    end

    # ② 역경로 캐시 (paths 불변이므로 1회만 계산)
    if isempty(layer._invpaths) || length(layer._invpaths) != Nz
        layer._invpaths = Vector{typeof(paths[1])}(undef, Nz)
        @inbounds for i in 1:Nz
            layer._invpaths[i] = reverse_paths(paths[i])
        end
    end
    invpaths = layer._invpaths

    # ③ Wadj 캐시 — CPU path에서만 필요
    if !is_gpu
        if length(layer._Wadj) != NW
            layer._Wadj = Vector{typeof(W[1])}(undef, NW)
            @inbounds for k in 1:NW
                layer._Wadj[k] = adjoint(W[k])
            end
        end
        Wadj = layer._Wadj
        @inbounds for k in 1:NW
            wsm = W[k].sm
            wadj_sm = Wadj[k].sm
            for j in 1:4
                for i in 1:4
                    wadj_sm[i,j] = conj(wsm[j,i])
                end
            end
        end
    end

    # ④ invhops + hop 출력 버퍼 캐시 (CPU path에서만 필요)
    if !is_gpu
        nt = nthreads()
        if isempty(layer._invhops) || length(layer._invhops) != nt
            layer._invhops = [Hopping_operator(layer.U, invpaths[1]) for _ in 1:nt]
            TF = typeof(z[1])
            layer._hop_out = [Vector{TF}([deepcopy(z[1])]) for _ in 1:nt]
        else
            @inbounds for t in 1:nt
                layer._invhops[t].Gauge = layer.U
            end
        end
    end

    # ⑤ 최적화: 먼저 c_i = Σ_j Wadj_{ij} * d_conj[j] 계산 후, hop 1회만 수행
    if z[1] isa WF3x1CPU
        ci_buffers = layer._ci_buffers
        invhops = layer._invhops
        hop_out = layer._hop_out

        @threads for i in 1:Nz
            tid = threadid()
            ci = ci_buffers[tid]
            zero_wf!(ci)
            base = (i-1) * Out
            @inbounds for j in 1:Out
                addmul!(ci, Wadj[base + j], d_conj[j])
            end
            invhop = invhops[tid]
            invhop.P = invpaths[i]
            invhop(ci, hop_out[tid])  # preallocated 버전 사용
            conj!(delta[i], hop_out[tid][1])
        end
    else
        # GPU batched: Wadj*conj(d) 를 1회 커널로 전부 계산
        NC = 3; NG = 4
        blocksize = z[1].blockinfo.blocksize
        rsize     = z[1].blockinfo.rsize

        # Wadj_gpu 동기화 (W_gpu가 있으면 GPU에서 직접, 없으면 CPU→GPU)
        if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
            layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        if layer._W_gpu !== nothing
            sync_Wadj_gpu!(layer._Wadj_gpu, layer._W_gpu)
        else
            # fallback: CPU Wadj → GPU
            Wadj_cpu = Array{ComplexF64}(undef, NG, NG, NW)
            @inbounds for k in 1:NW
                sm = Wadj[k].sm
                for jj in 1:4, ii in 1:4
                    Wadj_cpu[ii, jj, k] = sm[ii, jj]
                end
            end
            copyto!(layer._Wadj_gpu, Wadj_cpu)
        end

        # d_buf, ci_all: 공유 임시 버퍼 사용 (layers 간 재사용)
        _d_buf = get_shared_buf!(:d_buf, NC, NG, blocksize, rsize, Nd)
        pack_fermions!(_d_buf, d)

        _ci_buf = get_shared_buf!(:ci_all, NC, NG, blocksize, rsize, Nz)

        # 1회 batched kernel: ci_all = Σ_j Wadj * conj(d)
        batched_delta!(_ci_buf, layer._Wadj_gpu, _d_buf, blocksize, rsize, Nd, Nz)

        # lazy init inverse direction table for batched backward hop
        if layer._inv_hop_dirs_gpu === nothing
            max_steps = layer._hop_max_steps
            if max_steps == 0 && layer._hop_dirs_gpu === nothing
                # forward batched hop이 아직 init되지 않은 경우 — 여기서 직접 계산
                max_steps = 0
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        ns = 0
                    end
                    max_steps = max(max_steps, ns)
                end
                layer._hop_max_steps = max_steps

                # forward dirs + nsteps (forward에서 아직 init 안된 경우)
                dirs_cpu = zeros(Int8, max_steps, Nz)
                nsteps_cpu = zeros(Int32, Nz)
                for i in 1:Nz
                    steps_i = paths[i].various[1].steps
                    ns = length(steps_i)
                    if ns == 1 && steps_i[1].f == 0
                        nsteps_cpu[i] = Int32(0)
                    else
                        nsteps_cpu[i] = Int32(ns)
                        for s in 1:ns
                            dirs_cpu[s, i] = Int8(steps_i[s].f)
                        end
                    end
                end
                layer._hop_dirs_gpu   = CuArray(dirs_cpu)
                layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
            end

            max_steps = layer._hop_max_steps
            # inverse dirs: reverse order & negate
            inv_dirs_cpu = zeros(Int8, max_steps, Nz)
            for i in 1:Nz
                inv_steps = invpaths[i].various[1].steps
                ns = length(inv_steps)
                if ns == 1 && inv_steps[1].f == 0
                    # identity — nsteps=0
                else
                    for s in 1:ns
                        inv_dirs_cpu[s, i] = Int8(inv_steps[s].f)
                    end
                end
            end
            layer._inv_hop_dirs_gpu = CuArray(inv_dirs_cpu)
        end

        # batched inverse hop: ci_buf ↔ shared hop_temp
        max_steps = layer._hop_max_steps
        if max_steps > 0
            _hop_temp = get_shared_buf!(:hop_temp, NC, NG, blocksize, rsize, Nz)
            bi = z[1].blockinfo
            NX = z[1].NX; NY = z[1].NY; NZ_ = z[1].NZ; NT = z[1].NT
            batched_inv_hopping_gpu!(
                _ci_buf, _hop_temp, layer.U,
                layer._inv_hop_dirs_gpu, layer._hop_nsteps_gpu,
                max_steps, Nz, blocksize, rsize, bi, NX, NY, NZ_, NT)
        end

        # batch conj
        inplace_conj!(_ci_buf)

        # unpack → delta
        unpack_fermions!(delta, _ci_buf)
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
function DeltaRule(L::LossSumLayer,  ::Identity, z::Vector{<:WF3x1}, batch::Int)
    d1 = DeltaRule(L.l1, L.l1.activation, z, batch)
    d2 = DeltaRule(L.l2, L.l2.activation, z, batch)
    @assert length(d1) == length(d2) "DeltaRule outputs from loss layers must have the same length."
    result = similar(d1)
    for i in eachindex(d1)
        result[i] = similar(d1[i])
        axpby!(L.w1, d1[i], 0, result[i])
        axpby!(L.w2, d2[i], 1, result[i])
    end
    return result
end

function DeltaRule!(L::LossSumLayer, ::Identity, z::Vector{<:WF3x1}, delta::Vector, batch::Int)
    d = DeltaRule(L, L.activation, z, batch)
    @assert length(delta) == length(d) "DeltaRule! output buffer has wrong length."
    for i in eachindex(d)
        axpby!(1, d[i], 0, delta[i])
    end
    return delta
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


# function DeltaRule(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}) where T<:WF3x1



#     result = Vector{WF3x1}(undef, length(z))
#     α = layer.alpha


#     @inbounds for i in 1:length(z)
#         cz=conj(z[i])
#         norm = sqrt(norm2(z[i]))
#         eta1 = inner_product(cz, d[i])/2                  # z† d
#         eta2 = inner_product(z[i], conj(d[i]))/2            # zᵗ d*

#         result[i] = d[i] / norm
#         result[i] -= (eta1 * cz) / norm^3
#         result[i] -= (eta2 * cz) / norm^3
#     end

#     return result
# end

function DeltaRule!(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    n = length(z)
    if isempty(layer._conj_buf) || typeof(layer._conj_buf[1]) != typeof(z[1])
        layer._conj_buf = [_alloc_result_wf(z[1])]
    end
    cj_buf = layer._conj_buf[1]

    if n > 0 && z[1].f isa CuArray
        for i in 1:n
            _normlayer_bwd!(
                (delta[i]::WF3x1GPU).f,
                (z[i]::WF3x1GPU).f,
                (d[i]::WF3x1GPU).f)
        end
    else
        @inbounds for i in 1:n
            nrm   = sqrt(norm2(z[i]))
            inv_n = 1.0 / nrm

            conj!(cj_buf, z[i])
            eta1  = inner_product(cj_buf, d[i]) / 2
            coeff = -2 * real(eta1) * inv_n^3

            axpby!(inv_n, d[i], false, delta[i])
            axpby!(coeff, cj_buf, true,  delta[i])
        end
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

    q = layer.x  # forward에서 캐시된 직교화 벡터
    l = length(d)

    result = [deepcopy(d[i]) for i in 1:l]

    @inbounds for i in 1:l
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], result[i]) / denom
            axpby!(-coeff, q[j], 1, result[i])
        end
    end

    return result
end
function DeltaRule!(layer::GramSchmidt_Layer, ::Identity,
                   z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    q = layer.x  # forward에서 캐시된 직교화 벡터
    l = length(d)

    @inbounds for i in 1:l
        copyto!(delta[i].f, d[i].f)
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], delta[i]) / denom
            axpby!(-coeff, q[j], 1, delta[i])
        end
    end

end


# --- BlockScalar_Layer ---
function gradient(layer::BlockScalar_Layer, ::Identity, z::Vector{<:WF3x1}, d::Vector{T}, γ::Vector{Spin_matrix}) where T<:WF3x1
    if z[1] isa WF3x1GPU
        return [blockscalar_gradient_gpu!(layer, z, d)]
    end

    nblocks, n = size(layer.W)
    if size(layer._grad_W) != (nblocks, n)
        layer._grad_W = zeros(ComplexF64, nblocks, n)
    else
        fill!(layer._grad_W, zero(ComplexF64))
    end
    grad_W = layer._grad_W

    if false
        # GPU: preallocated buffers로 사이트별 내적 → 블록별 합산 (alloc 0)
        blockscalar_gradient_gpu!(grad_W, layer, z, d)
        return grad_W
        bmap = layer._gpu_block_map
        nblocks_total = length(bmap)
        fref = d[1].f  # CuArray{ComplexF64,4} reference

        # 1회만 버퍼 초기화
        if layer._grad_prod_buf === nothing || size(layer._grad_prod_buf) != size(fref)
            layer._grad_prod_buf = CUDA.zeros(ComplexF64, size(fref)...)
            layer._grad_site_buf = CUDA.zeros(ComplexF64, 1, 1, size(fref,3), size(fref,4))
            layer._grad_site_cpu = Vector{ComplexF64}(undef, nblocks_total)
        end
        prod_buf = layer._grad_prod_buf
        site_buf = layer._grad_site_buf

        for k in 1:n
            prod_buf .= d[k].f .* z[k].f
            _sum_spin_color!(site_buf, prod_buf)
            copyto!(layer._grad_site_cpu, vec(site_buf))

            @inbounds for i in 1:nblocks_total
                grad_W[bmap[i], k] += conj(layer._grad_site_cpu[i])
            end
        end
    else
        # CPU: block_map으로 직접 내적 (blockize 제거)
        NC, NG = z[1].NC, z[1].NG
        NX, NY, NZ, NT = z[1].NX, z[1].NY, z[1].NZ, z[1].NT
        bmap = layer._block_map

        for k in 1:n
            zk = z[k]; dk = d[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                s = zero(ComplexF64)
                for ig in 1:NG
                    @simd for ic in 1:NC
                        # inner_product(conj(d), z) = Σ d*z, then conj → ∂L/∂W*
                        s += dk[ic, ix, iy, iz, it, ig] * zk[ic, ix, iy, iz, it, ig]
                    end
                end
                grad_W[bid, k] += conj(s)
            end
        end
    end

    return grad_W
end

function DeltaRule(layer::BlockScalar_Layer, ::Identity,
                   z::Vector{<:WF3x1}, d::Vector{T}) where T<:WF3x1
    n = length(d)
    nblocks = size(layer.W, 1)
    r = layer.r

    # 출력 버퍼 캐시 (deepcopy 1회만, 또는 GPU↔CPU 전환 시 재초기화)
    if !layer._initialized || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(d[1]))
        layer._result = Vector{WF3x1}(undef, n)
        for k in 1:n
            layer._result[k] = deepcopy(d[k])
        end
        if d[1] isa WF3x1GPU
            _init_gpu_block_cache!(layer, d[1])
        elseif isempty(layer._block_map)
            NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
            NBX = NX÷r; NBY = NY÷r; NBZ = NZ÷r; NBT = NT÷r
            nsites = NX * NY * NZ * NT
            layer._block_map = Vector{Int}(undef, nsites)
            idx = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                idx += 1
                bx = (ix-1)÷r + 1; by = (iy-1)÷r + 1
                bz = (iz-1)÷r + 1; bt = (it-1)÷r + 1
                layer._block_map[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
            end
        end
        layer._initialized = true
    end
    result = layer._result

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(result[k].f, d[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        # CPU: block_map으로 직접 스칼라 곱 (blockize/unblockize 제거)
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            rk = result[k]
            dk = d[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        rk[ic, ix, iy, iz, it, ig] = w * dk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, dk, 1, rk)
            end
        end
    end
    return result
end

function DeltaRule!(layer::BlockScalar_Layer, ::Identity,
                    z::Vector{<:WF3x1}, d::Vector{T}, delta::Vector{<:WF3x1}) where T<:WF3x1
    n = length(d)
    nblocks = size(layer.W, 1)
    r = layer.r

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(delta[k].f, d[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        # CPU: block_map으로 직접 스칼라 곱 (blockize/unblockize 제거)
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            dk_out = delta[k]
            dk = d[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        dk_out[ic, ix, iy, iz, it, ig] = w * dk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, dk, 1, dk_out)
            end
        end
    end
end

# --- BlockMatrix_Layer ---
function gradient(layer::BlockMatrix_Layer, ::Identity, z::Vector{<:WF3x1}, d::Vector{T}, γ::Vector{Spin_matrix}) where T<:WF3x1
    if z[1] isa WF3x1GPU
        return [blockmatrix_gradient_gpu!(layer, z, d)]
    end

    nblocks, nout, nin = size(layer.W)
    length(z) == nin || error("BlockMatrix_Layer gradient expected $nin input vectors, got $(length(z))")
    length(d) == nout || error("BlockMatrix_Layer gradient expected $nout output deltas, got $(length(d))")
    if size(layer._grad_W) != (nblocks, nout, nin)
        layer._grad_W = zeros(ComplexF64, nblocks, nout, nin)
    else
        fill!(layer._grad_W, zero(ComplexF64))
    end
    grad_W = layer._grad_W

    NC, NG = z[1].NC, z[1].NG
    NX, NY, NZ, NT = z[1].NX, z[1].NY, z[1].NZ, z[1].NT
    bmap = layer._block_map

    @inbounds for o in 1:nout
        do_ = d[o]
        for i in 1:nin
            zi = z[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                s = zero(ComplexF64)
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s += do_[ic, ix, iy, iz, it, ig] * zi[ic, ix, iy, iz, it, ig]
                    end
                end
                grad_W[bid, o, i] += conj(s)
            end
        end
    end

    return grad_W
end

function DeltaRule(layer::BlockMatrix_Layer, ::Identity,
                   z::Vector{<:WF3x1}, d::Vector{T}) where T<:WF3x1
    nblocks, nout, nin = size(layer.W)
    length(d) == nout || error("BlockMatrix_Layer DeltaRule expected $nout output deltas, got $(length(d))")
    r = layer.r

    if !layer._initialized || length(layer._result) != nin || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(d[1]))
        layer._result = Vector{WF3x1}(undef, nin)
        for i in 1:nin
            layer._result[i] = deepcopy(d[min(i, nout)])
        end
        if d[1] isa WF3x1GPU
            _init_gpu_block_cache!(layer, d[1])
        elseif isempty(layer._block_map)
            NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
            NBX = NX÷r; NBY = NY÷r; NBZ = NZ÷r; NBT = NT÷r
            nsites = NX * NY * NZ * NT
            layer._block_map = Vector{Int}(undef, nsites)
            idx = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                idx += 1
                bx = (ix-1)÷r + 1; by = (iy-1)÷r + 1
                bz = (iz-1)÷r + 1; bt = (it-1)÷r + 1
                layer._block_map[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
            end
        end
        layer._initialized = true
    end
    result = layer._result

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockmatrix_W_gpu!(layer; upload=false)
        blockmatrix_delta_gpu!(result, d, W_gpu, bmap_gpu, layer.residual)
    else
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        @inbounds for i in 1:nin
            ri = result[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s = zero(ComplexF64)
                        for o in 1:nout
                            s += W[bid, o, i] * d[o][ic, ix, iy, iz, it, ig]
                        end
                        if layer.residual && i <= nout
                            s += d[i][ic, ix, iy, iz, it, ig]
                        end
                        ri[ic, ix, iy, iz, it, ig] = s
                    end
                end
            end
        end
    end
    return result
end

function DeltaRule!(layer::BlockMatrix_Layer, ::Identity,
                    z::Vector{<:WF3x1}, d::Vector{T}, delta::Vector{<:WF3x1}) where T<:WF3x1
    nblocks, nout, nin = size(layer.W)
    length(delta) == nin || error("BlockMatrix_Layer DeltaRule! expected $nin input delta buffers, got $(length(delta))")
    length(d) == nout || error("BlockMatrix_Layer DeltaRule! expected $nout output deltas, got $(length(d))")

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockmatrix_W_gpu!(layer; upload=false)
        blockmatrix_delta_gpu!(delta, d, W_gpu, bmap_gpu, layer.residual)
    else
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        @inbounds for i in 1:nin
            ri = delta[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s = zero(ComplexF64)
                        for o in 1:nout
                            s += W[bid, o, i] * d[o][ic, ix, iy, iz, it, ig]
                        end
                        if layer.residual && i <= nout
                            s += d[i][ic, ix, iy, iz, it, ig]
                        end
                        ri[ic, ix, iy, iz, it, ig] = s
                    end
                end
            end
        end
    end
end


end # module activation

