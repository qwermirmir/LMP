include("spin_color_vector.jl")





function Lossfunction(f::spin_color_vector,t::spin_color_vector)

    loss=f-t
    
    result = inner_product(loss,loss)
    return result/2

end


struct PTC_Layer
    W::Vector{Spin_matrix}  # 가중치 행렬 (Spin_matrix 리스트)
    hop::Hopping_operator   # Hopping 연산자
    activation::Function    # 활성화 함수 (선택적)

    function PTC_Layer(W::Vector{Spin_matrix}, hop::Hopping_operator; activation::Function = identity)
        new(W, hop, activation)
    end
end

function (layer::PTC_Layer)(x::WF3x1)
    result = deepcopy(x)
    fields = layer.hop(x)  # Hopping operator를 적용한 field들

    if length(layer.W) == 1
        # 가중치가 하나뿐인 경우 (모든 field에 같은 가중치를 곱함)
        for i in 1:length(fields)
            result += layer.W[1] * fields[i]
        end
    elseif length(layer.W) == length(fields)
        # 가중치 벡터와 field 벡터 크기가 같은 경우
        for i in 1:length(fields)
            result += layer.W[i] * fields[i]
        end
    else
        error("W의 길이가 1이거나 fields와 같아야 합니다.")
    end

    return layer.activation(result)
end

function extend_layer(prev_layer::PTC_Layer, new_W::Vector{Spin_matrix}, new_hop::Hopping_operator; activation::Function = identity)
    return PTC_Layer(new_W, new_hop, activation=activation)
end





















