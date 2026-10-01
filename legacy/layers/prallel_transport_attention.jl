# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct PrallelTransportAttention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool
    bias::Float64

    function PrallelTransportAttention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,  activation::ActivationFunction=Identity(), residual::Bool=false , bias::Float64=0.1)
        new(Q, K, V, nominator,  activation,  residual,bias)
    end

    function PrallelTransportAttention_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false , bias::Float64=0.1)
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV

        new(Q, K, V, nominator,  activation,  residual, bias)
    end
end

_has_gpu(layer::PrallelTransportAttention_Layer) = _has_gpu(layer.Q)

# --- PrallelTransportAttention_Layer ---
function _to_cpu(layer::PrallelTransportAttention_Layer)
    return PrallelTransportAttention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
end

# --- PrallelTransportAttention_Layer ---
function _to_gpu(layer::PrallelTransportAttention_Layer, blocks=nothing)
    return PrallelTransportAttention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
end

function _to_gpu(layer::PrallelTransportAttention_Layer, blocks, shared_U)
    return PrallelTransportAttention_Layer(_to_gpu(layer.Q, blocks, shared_U), _to_gpu(layer.K, blocks, shared_U), _to_gpu(layer.V, blocks, shared_U),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
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

 ratio = zeros(Float64, 4) #



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

function update!(layer::PrallelTransportAttention_Layer, opt::Adam, grad )
    param = layer.K.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
        end
    end


    param = layer.V.W
    if length(param) ==1
        complexProcess(param, opt , grad[2])  
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[2][i])
        end
    end

end 
