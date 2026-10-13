# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct FullPTCAttention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool

    function FullPTCAttention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,    activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,  activation,  residual)
    end

    function FullPTCAttention_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator,  activation,  residual)
    end
end

_has_gpu(layer::FullPTCAttention_Layer) = _has_gpu(layer.Q)

# --- FullPTCAttention_Layer ---
function _to_cpu(layer::FullPTCAttention_Layer)
    return FullPTCAttention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                                  layer.nominator, layer.activation, layer.residual)
end

# --- FullPTCAttention_Layer ---
function _to_gpu(layer::FullPTCAttention_Layer, blocks=nothing)
    return FullPTCAttention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                                  layer.nominator, layer.activation, layer.residual)
end

function _to_gpu(layer::FullPTCAttention_Layer, blocks, shared_U)
    return FullPTCAttention_Layer(_to_gpu(layer.Q, blocks, shared_U), _to_gpu(layer.K, blocks, shared_U), _to_gpu(layer.V, blocks, shared_U),
                                  layer.nominator, layer.activation, layer.residual)
end

function (layer::FullPTCAttention_Layer)(x::WF3x1)


    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W




    lW=length(KW)


    m=layer.nominator


    Hx=layer.V.hop(x)


    # if Q.W === nothing || K.W === nothing || V.W === nothing
 # error("gradient() Q, K, V W .")
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

function update!(layer::FullPTCAttention_Layer, opt::Adam, grad )
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
        complexProcess(param, opt , grad[3])  
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[3][i])
        end
    end

    


end
