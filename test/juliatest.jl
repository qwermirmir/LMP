using CUDA
using LinearAlgebra
using Random
using LatticeDiracOperators
using InteractiveUtils
using Gaugefields
import Gaugefields.AbstractGaugefields_module:getvalue,setvalue!

println("Test for 1/D for Wilson CUDA")

versioninfo()

version = Base.pkgversion(Gaugefields)
println("version of Gaugefields : ", version)
version = Base.pkgversion(LatticeDiracOperators)
println("version of LatticeDiracOperators: ", version)

const pes =  (2, 2, 2, 2)
const cuda = true
const numtrj_in = 2 # the number of operations to take average of performance
const MDsteps_in = 0 # dummy

const βin = 6.0
const NXin = 8
const NYin = 8
const NZin = 8
const NTin = 16
const κin = 0.141139

header = """
Parameters
const blocks = $pes
const cuda = $cuda
const numtrj_in = $numtrj_in
const MDsteps_in = $MDsteps_in

const βin = $βin
const NXin = $NXin
const NYin = $NYin
const NZin = $NZin
const NTin = $NTin
const κin = $κin
"""

function test1()
    NX = NXin
    NY = NYin
    NZ = NZin
    NT = NTin
    Nwing = 0
    Dim = 4
    NC = 3

    if cuda
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",cuda=true,blocks=PEs)

    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    end
    Ucpu = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    substitute_U!(Ucpu,U)

    Random.seed!(123+get_myrank(U[1]))

    if get_myrank(U) == 0
        println(header) # print header
    end

    snet = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = βin/2
    push!(snet,β,plaqloop)

    #show(snet)

    Random.seed!(123)
    x = Initialize_pseudofermion_fields(U[1],"Wilson",nowing=true)
    xcpu = Initialize_pseudofermion_fields(Ucpu[1],"Wilson",nowing=true)

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κin
    params["faster version"] = true
    params["eps_CG"] = 1.0e-16
    params["verbose_level"] = 3 # to show CG counts

    D = Dirac_operator(U,x,params)
    Dcpu = Dirac_operator(Ucpu,xcpu,params)
    gauss_distribution_fermion!(xcpu)

    substitute_fermion!(x,xcpu)


    b = similar(x)
    y = similar(x)
    bcpu = similar(xcpu)
    ycpu = similar(xcpu)
    @time mul!(b,D,x)
    @time mul!(bcpu,Dcpu,xcpu)


    println_verbose_level1(U[1],"Benchmark!!")
    @time solve_DinvX!(y,D,b)
    @time solve_DinvX!(ycpu,Dcpu,bcpu)



    return
end
test1()