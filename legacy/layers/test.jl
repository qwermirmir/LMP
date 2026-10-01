# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

mutable struct test_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Float64
    test::WF3x1
    activation::ActivationFunction
    residual::Bool

    function test_Layer(Q::Layer, K::Layer, V::Layer, nominator::Float64 ,  test::WF3x1,  activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator, test , activation,  residual)
    end

    function test_Layer(Q::Layer, nominator::Float64 ,test::WF3x1, activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator, test, activation,  residual)
    end
end

_has_gpu(layer::test_Layer) = _has_gpu(layer.Q)

# --- test_Layer ---
function _to_cpu(layer::test_Layer)
    layer.Q = _to_cpu(layer.Q)
    layer.K = _to_cpu(layer.K)
    layer.V = _to_cpu(layer.V)
    if layer.test isa WF3x1GPU
        layer.test = _to_cpu_fermion(layer.test)
    end
    return layer
end

# --- test_Layer ---
function _to_gpu(layer::test_Layer, blocks=nothing)
    layer.Q = _to_gpu(layer.Q, blocks)
    layer.K = _to_gpu(layer.K, blocks)
    layer.V = _to_gpu(layer.V, blocks)
    if layer.test isa WF3x1CPU
        layer.test = _to_gpu_fermion(layer.test; blocks=blocks)
    end
    return layer
end

function _to_gpu(layer::test_Layer, blocks, shared_U)
    layer.Q = _to_gpu(layer.Q, blocks, shared_U)
    layer.K = _to_gpu(layer.K, blocks, shared_U)
    layer.V = _to_gpu(layer.V, blocks, shared_U)
    if layer.test isa WF3x1CPU
        layer.test = _to_gpu_fermion(layer.test; blocks=blocks)
    end
    return layer
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
 # error("gradient() Q, K, V W .")
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

function update!(layer::test_Layer, opt::Adam, grad )
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
        realProcess(param, opt , grad[2])  
    else 
        for i in 1:length(param)
            realProcess(param[i], opt , grad[2][i])
        end
    end

    


end
