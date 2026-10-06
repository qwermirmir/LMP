# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct Attention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    
    residual::Bool

    function Attention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number , activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,activation, residual)
    end

    function Attention_Layer(Q::Layer, nominator::Number , activation::ActivationFunction=Identity(), residual::Bool=false)
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator, activation, residual)
    end
end

_has_gpu(layer::Attention_Layer) = _has_gpu(layer.Q)

# --- Attention_Layer ---
function _to_cpu(layer::Attention_Layer)
    return Attention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                           layer.nominator, layer.activation, layer.residual)
end

# --- Attention_Layer ---
function _to_gpu(layer::Attention_Layer, blocks=nothing)
    return Attention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                           layer.nominator, layer.activation, layer.residual)
end

function _to_gpu(layer::Attention_Layer, blocks, shared_U)
    return Attention_Layer(_to_gpu(layer.Q, blocks, shared_U), _to_gpu(layer.K, blocks, shared_U), _to_gpu(layer.V, blocks, shared_U),
                           layer.nominator, layer.activation, layer.residual)
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

function update!(layer::Attention_Layer, opt::Adam, grad )
    if layer.Q isa PTC_Layer
 

        param = layer.Q.W
        if length(param) ==1
            complexProcess(param, opt , grad[1] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[1][i] )
            end
        end

        param = layer.K.W
        if length(param) ==1
            complexProcess(param, opt , grad[2] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[2][i] )
            end
        end

        param = layer.V.W
        if length(param) ==1
            complexProcess(param, opt , grad[3] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[3][i] )
            end
        end
    end


    if layer.Q isa CPTC_Layer
        param = layer.Q.W
        if length(param) ==1
            realProcess(param, opt , grad[1])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[1][i])
            end
        end
    
        param = layer.K.W
        if length(param) ==1
            realProcess(param, opt , grad[2])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[2][i])
            end
        end
    
        param = layer.V.W
        if length(param) ==1
            realProcess(param, opt , grad[3])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[3][i])
            end
        end
    end
    


end
