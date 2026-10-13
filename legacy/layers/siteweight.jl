# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct siteweight_Layer<: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool

    function siteweight_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,  activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,  activation,  residual)
    end

    function siteweight_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator,  activation,  residual)
    end
end

_has_gpu(layer::siteweight_Layer) = _has_gpu(layer.Q)

# --- siteweight_Layer ---
function _to_cpu(layer::siteweight_Layer)
    return siteweight_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                            layer.nominator, layer.activation, layer.residual)
end

# --- siteweight_Layer ---
function _to_gpu(layer::siteweight_Layer, blocks=nothing)
    return siteweight_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                            layer.nominator, layer.activation, layer.residual)
end

function _to_gpu(layer::siteweight_Layer, blocks, shared_U)
    return siteweight_Layer(_to_gpu(layer.Q, blocks, shared_U), _to_gpu(layer.K, blocks, shared_U), _to_gpu(layer.V, blocks, shared_U),
                            layer.nominator, layer.activation, layer.residual)
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

function update!(layer::siteweight_Layer, opt::Adam, grad)
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
