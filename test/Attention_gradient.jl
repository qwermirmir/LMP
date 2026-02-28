using LatticeDiracOperators
using LatticeQCD
using Gaugefields
using Wilsonloop
# using LMP




NX = 8
NY = 8
NZ = 8
NT = 16
Nwing = 0
Dim = 4
NC = 3
U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
x = Initialize_pseudofermion_fields(U[1],"Wilson",nowing=true)
gauss_distribution_fermion!(x)
t = Initialize_pseudofermion_fields(U[1],"Wilson",nowing=true)
gauss_distribution_fermion!(t)
x=x/sqrt(norm2(x))
t=t/sqrt(norm2(t))





delta=1e-8
P=Onestep()
hop=Hopping_operator(U,P)
W=Generate_weight(hop)


layer=CPTC_Layer(hop,W)
act=Softmax()
Attention=Attention_Layer(layer,act)
layers=[Attention]
y=layers(x)
f=norm2(y-t)
grad1=deepcopy(Attention.Q.W)
grad2=deepcopy(Attention.K.W)
grad3=deepcopy(Attention.V.W)




for i in 1:9
    for u in 1:4
        for v in 1:4
            local layer1=deepcopy(layers)
            layer1[1].Q.W[i][u,v]+=delta
            y1=layer1(x)
            grad1[i][u,v]=(norm2(y1-t)-f)/delta
        end
    end
end 

for i in 1:9
    for u in 1:4
        for v in 1:4
            local layer1=deepcopy(layers)
            layer1[1].K.W[i][u,v]+=delta
            y1=layer1(x)
            grad2[i][u,v]=(norm2(y1-t)-f)/delta
        end
    end
end 

for i in 1:9
    for u in 1:4
        for v in 1:4
            local layer1=deepcopy(layers)
            layer1[1].V.W[i][u,v]+=delta
            y1=layer1(x)
            grad3[i][u,v]=(norm2(y1-t)-f)/delta
        end
    end
end 









u = forward_propagation(layers, x)
delta = backward_propagation(layers, t, u)



d = gradient(layers[1], layers[1].activation::ActivationFunction, u[1], delta,γ)


println(grad1[1]-d[1][1])
println(grad2[1]-d[2][1])
println(grad3[1]-d[3][1])
    










