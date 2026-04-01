



subgrad=similar(W)

for i in 1:9



    for u in 1:4
        for v in 1:4
            W2=deepcopy(W)
            W3=deepcopy(W)
            delta = 1e-9
            W2[i][u, v] = W2[i][(u-1) * 4 + v]+delta
            W3[i][u, v] = W3[i][(u-1) * 4 + v]-delta
            layer2 = CPTC_Layer(hop, W2)
            layer3 = CPTC_Layer(hop, W3)
            L=(Lossfunction(layer2(x),t)-Lossfunction(layer3(x),t))/(2*delta)

            subgrad[i][u, v]=L



        end
    end

end 









W=Generate_weight(hop)
W=0.001*W
W2=deepcopy(W)
delta = 1e-9
W2[i][1,1]=W2[i][1,1]+delta
W3=deepcopy(W)
W3[i][1,1]=W3[i][1,1]-delta
layer1 = CPTC_Layer(hop, W)
layer2 = CPTC_Layer(hop, W2)
layer3 = CPTC_Layer(hop, W3)


L=(Lossfunction(layer2(x),t)-Lossfunction(layer3(x),t))/(2*delta)
sequence=[layer1]


adam=Adam()


layers=[layer1]
result, u = forward_propagation([layer1], x)
b= backward_propagation([layer1], x, t, u)
d = gradient(layers[1], layers[1].activation, u[2], b[1])
println(L-d[i][1,1])




