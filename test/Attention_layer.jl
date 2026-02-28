using LatticeDiracOperators
using LatticeQCD
using Gaugefields
using Wilsonloop
using LMP
using LinearAlgebra
using Random







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



params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.141139
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2

D = Dirac_operator(U,x,params)



P=Onestep()
hop=Hopping_operator(U,P)
W=Generate_weight(hop)




layer1=CPTC_Layer(hop,W)


act=Softmax()
Attention=Attention_Layer(layer1,act)
layers=[Attention]
opt=Adam()
opt.α=0.05
train(opt,[layer1],x,t,D,epochs=200)





