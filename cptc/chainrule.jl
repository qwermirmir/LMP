include("spin_color_vector.jl")





function Lossfunction(f::spin_color_vector,t::spin_color_vector)

    loss=f-t
    
    result = inner_product(loss,loss)
    return result/2

end




function build_layers()
    # 사용자에게 허용할 레이어 타입(함수) 목록 예시
    type_list = ["ptc", "Dense", "Convolution", "Activation"]

    layers = String[]   # 최종적으로 레이어 정보를 담을 배열
    layer_idx = 1       # 몇 번째 레이어인지 표시
    building_layers_check = true

    println("==== 머신러닝 레이어 구성 시작 ====")

    while building_layers_check
        println("레이어 $layer_idx 의 '기본 타입'을 입력하세요 (예: Dense, Activation, Convolution 등).")
        println("종료하려면 'done'을 입력하세요.")
        print(">> ")
        layer_type = readline()  # 사용자가 입력할 때까지 대기

        # 사용자가 "done"을 입력하면 최종 종료
        if layer_type == "done"
            println("레이어 입력을 종료합니다.")
            building_layers_check = false

        elseif isempty(layer_type)
            println("입력값이 비어 있습니다. 다시 입력해주세요.")

        else
            # hidden 배열 안에, 이 레이어를 구성하는 여러 함수(타입)을 담는다고 가정
            hidden = String[]

            # 우선 현재 입력한 layer_type이 우리가 아는 타입인지 확인
            # - 만약 알 수 없는 타입이면 재입력 받도록 한다.
            while !(layer_type in type_list)
                if layer_type == "done"
                    # 만약 "done"이면, 그냥 이 레이어 구성 자체를 종료하겠다고 가정
                    println("현재 레이어 구성 종료.")
                    break
                end
                println("그러한 함수(`$layer_type`)는 발견되지 않았습니다. 다시 입력해주세요.")
                layer_type = readline()

                # 비어 있으면 계속 재요청
                if isempty(layer_type)
                    println("입력값이 비어 있습니다. 다시 입력해주세요.")
                end
            end

            # 만약 valid한 type_list 안에 들어있으면 hidden에 추가
            if (layer_type in type_list)
                push!(hidden, layer_type)
            end

            # 추가로, "이 레이어 안에 더 많은 함수(서브레이어 등)를 붙일 것인지" 확인하는 루프
            # 여기서는 "done"을 입력하면 서브레이어 입력 종료
            while true
                println("계속 다른 함수를 추가하려면 입력하고, 이 레이어 구성을 마치려면 'done'을 입력하세요.")
                println("현재 구성된 서브레이어들: $hidden")
                print(">> ")
                hlayer_type = readline()

                if hlayer_type == "done"
                    # 서브레이어(함수) 구성을 종료
                    break
                elseif isempty(hlayer_type)
                    println("입력값이 비어 있습니다. 다시 입력해주세요.")
                else
                    # 유효한 타입인지 확인
                    if hlayer_type in type_list
                        push!(hidden, hlayer_type)
                    else
                        println("`$hlayer_type`은(는) type_list에 없습니다.")
                    end
                end
            end

            # 이제 hidden 배열에 모인 "함수/타입들"을 하나의 레이어로 간주해 layers에 push
            # 여기서는 예시로, hidden 내용을 쉼표로 합쳐서 문자열로 만든다.
            push!(layers, join(hidden, ", "))

            println("현재 레이어 구성:", layers)
            layer_idx += 1
        end
    end

    println("==== 최종 레이어 구성 ====")
    println("최종 레이어 : $layers")

    return layers
end




















