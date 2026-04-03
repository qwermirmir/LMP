using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using LMP

"""
Gauge covariance test for Hopping_operator:
    hop_V(g·ψ) == g · hop_U(ψ)
where V_μ(x) = g(x) U_μ(x) g†(x+μ̂)
"""
function test_hopping_covariance(; NX=4, NY=4, NZ=4, NT=4, NDW=0, use_gpu=false)
    NC = 3
    
    # ---------- gauge field ----------
    if use_gpu
        PEs = (2, 2, 2, 2)
        U = Initialize_Gaugefields(NC, NDW, NX, NY, NZ, NT,
                condition="hot", cuda=true, blocks=PEs)
        g_field = Initialize_Gaugefields(NC, NDW, NX, NY, NZ, NT,
                condition="hot", cuda=true, blocks=PEs)
        temp = similar(g_field[1])
    else
        U = Initialize_Gaugefields(NC, NDW, NX, NY, NZ, NT, condition="hot")
        g_field = Initialize_Gaugefields(NC, NDW, NX, NY, NZ, NT, condition="hot")
        temp = similar(g_field[1])
    end

    g = g_field[1]   # 단일 gauge transformation 행렬 g(x)
    
    # ---------- fermion field ----------
    x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
    gauss_distribution_fermion!(x)
    
    # ---------- gauge transform: V_μ = g U_μ g†(x+μ̂) ----------
    V = deepcopy(U)
    gUgshift!(V, g, temp)
    
    # ---------- g·ψ ----------
    gx = deepcopy(x)
    mul!(gx, g, x)
    
    # ---------- test all directions ----------
    directions = Dict(
        "+1" => Paths([1]),
        "+2" => Paths([2]),
        "+3" => Paths([3]),
        "+4" => Paths([4]),
        "-1" => Paths([-1]),
        "-2" => Paths([-2]),
        "-3" => Paths([-3]),
        "-4" => Paths([-4]),
    )
    
    println("=" ^ 60)
    println("Hopping_operator gauge covariance test")
    println("  NDW=$NDW  GPU=$use_gpu  lattice=$(NX)×$(NY)×$(NZ)×$(NT)")
    println("=" ^ 60)
    
    all_pass = true
    for (name, P) in sort(collect(directions), by=first)
        hop_U = Hopping_operator(U, P)
        hop_V = Hopping_operator(V, P)
        
        # g · hop_U(ψ)
        lhs = deepcopy(hop_U(x)[1])
        mul!(lhs, g, hop_U(x)[1])
        
        # hop_V(g·ψ)
        rhs = hop_V(gx)[1]
        
        diff = real(norm2(lhs - rhs))
        norm_lhs = real(norm2(lhs))
        rel = diff / max(norm_lhs, 1e-30)
        
        pass = rel < 1e-12
        all_pass &= pass
        status = pass ? "PASS" : "FAIL"
        println("  direction $name:  |g·hop_U(ψ) - hop_V(g·ψ)|² / |g·hop_U(ψ)|² = $(rel)  [$status]")
    end
    
    # ---------- multi-step path test ----------
    println("-" ^ 60)
    println("Multi-step path test: [+1, +2]")
    P_multi = Paths([1, 2])
    hop_U = Hopping_operator(U, P_multi)
    hop_V = Hopping_operator(V, P_multi)
    
    lhs = deepcopy(hop_U(x)[1])
    mul!(lhs, g, hop_U(x)[1])
    rhs = hop_V(gx)[1]
    
    diff = real(norm2(lhs - rhs))
    norm_lhs = real(norm2(lhs))
    rel = diff / max(norm_lhs, 1e-30)
    pass = rel < 1e-12
    all_pass &= pass
    status = pass ? "PASS" : "FAIL"
    println("  path [+1,+2]:  rel_diff = $(rel)  [$status]")
    
    println("=" ^ 60)
    println(all_pass ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗")
    println("=" ^ 60)
    return all_pass
end

# --- nowing (NDW=0) ---
println("\n>>> Test: NDW=0 (nowing) <<<\n")
test_hopping_covariance(NDW=0)
