using LatticeDiracOperators
using Gaugefields
using Wilsonloop
using LMP
using JLD2

U = load("C://Users/_s2220186/Desktop/Julia/gauge/Q0.jld2", "U")
x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
gauss_distribution_fermion!(x)
x = x / sqrt(norm2(x))

params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.1605
params["eps_CG"] = 1.0e-20
params["verbose_level"] = 0
D = Dirac_operator(U, x, params)

model = load("C:\\Users\\_s2220186\\Desktop\\multilowmode\\MG-testvector\\32PTLayerRitz(mass016)(30000epoch)(i=10)(4eig)(60config).jld2", "layers")
for layer in model
    if layer isa PTLayer
        layer.U = U
    end
end
model = model[1:end-1]
v = model([U])
v_before = deepcopy(v)

setup = setup_DDalphaAMG(D, x, 4, 4; nsmooth=16, verbose=false, coarse_vectors=v)

maxdiff = 0.0
for i in eachindex(v)
    diff = sqrt(norm2(v[i] - v_before[i]))
    maxdiff = max(maxdiff, diff)
end

println("external_v_maxdiff_after_setup=", maxdiff)
println("setup_uses_distinct_storage=", !(setup.test_vectors === v))
