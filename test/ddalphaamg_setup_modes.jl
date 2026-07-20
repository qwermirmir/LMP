using Test
using LMP
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra

const DDAMG = LMP.DDalphaAMG

@testset "DDalphaAMG setup mode validation" begin
    @test DDAMG._amg_validate_setup_mode(:inverse_sap) == :inverse_sap
    @test DDAMG._amg_validate_setup_mode(:direct_near_null) == :direct_near_null
    @test_throws ArgumentError DDAMG._amg_validate_setup_mode(:unknown)
end

function make_setup_mode_cpu_problem()
    U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="cold", cuda=false)
    x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
    params = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.12,
        "faster version" => true,
        "verbose_level" => 0,
    )
    return x, Dirac_operator(U, x, params)
end

field_distance(a, b) = sqrt(max(0.0,
    real(inner_product(a, a)) + real(inner_product(b, b)) -
    2real(inner_product(a, b))))

@testset "direct near-null external phase uses homogeneous SAP" begin
    template, D = make_setup_mode_cpu_problem()
    seed = similar(template)
    DDAMG.random_fermion!(seed)

    expected = DDAMG._copy_wf(seed)
    zero_rhs = similar(template)
    DDAMG.zero_wf!(zero_rhs)
    reference_setup = DDAMG.DDalphaAMG_Setup{typeof(template)}(
        [expected], 2, 1,
        template.NX, template.NY, template.NZ, template.NT,
        template.NC, template.NG,
        2, 2, 2, 2, 16, nothing, nothing,
        DDAMG.CoarseVector[], nothing, nothing,
        zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing)
    @test reference_setup.setup_mode == :inverse_sap
    DDAMG.sap_smooth!(expected, D, zero_rhs, reference_setup;
        niter=1,
        block_solver=:mr,
        nblock_mr=1,
        odd_even=false,
        sap_schedule=:twocolor)

    setup = DDAMG.setup_DDalphaAMG(D, template, 2, 1;
        setup_mode=:direct_near_null,
        nsmooth=1,
        verbose=false,
        coarse_vectors=[seed],
        smooth_external_vectors=true,
        setup_smoother=:sap,
        setup_sap_iters=1,
        setup_block_solver=:mr,
        setup_block_mr_steps=1,
        sap_schedule=:twocolor,
        use_chirality=false,
        precompute_coarse=false,
        block_normalize=false)

    @test setup.setup_mode == :direct_near_null
    @test field_distance(setup.test_vectors[1], expected) < 1e-12
end


@testset "omitted setup mode preserves inverse-SAP setup" begin
    template, D = make_setup_mode_cpu_problem()
    seed = similar(template)
    DDAMG.random_fermion!(seed)
    common = (
        nsmooth=1,
        verbose=false,
        coarse_vectors=[seed],
        smooth_external_vectors=true,
        setup_smoother=:mr,
        use_chirality=false,
        precompute_coarse=false,
        block_normalize=false,
    )

    implicit = DDAMG.setup_DDalphaAMG(D, template, 2, 1; common...)
    explicit = DDAMG.setup_DDalphaAMG(D, template, 2, 1;
        setup_mode=:inverse_sap, common...)

    @test implicit.setup_mode == :inverse_sap
    @test explicit.setup_mode == :inverse_sap
    @test field_distance(implicit.test_vectors[1], explicit.test_vectors[1]) < 1e-12
end

@testset "direct near-null external vectors bypass random adaptive loop" begin
    template, D = make_setup_mode_cpu_problem()
    seed = similar(template)
    DDAMG.random_fermion!(seed)
    common = (
        setup_mode=:direct_near_null,
        nsmooth=0,
        verbose=false,
        coarse_vectors=[seed],
        use_chirality=false,
        precompute_coarse=false,
        block_normalize=false,
    )

    without_adaptation = DDAMG.setup_DDalphaAMG(
        D, template, 2, 1; nvcycle=0, common...)
    with_ignored_adaptation = DDAMG.setup_DDalphaAMG(
        D, template, 2, 1; nvcycle=6, common...)

    @test field_distance(
        without_adaptation.test_vectors[1],
        with_ignored_adaptation.test_vectors[1]) < 1e-12
end

@testset "direct near-null adaptive update preserves historical order" begin
    template, D = make_setup_mode_cpu_problem()
    prototype = similar(template)
    DDAMG.random_fermion!(prototype)
    coarse_setup = DDAMG.setup_DDalphaAMG(D, template, 2, 1;
        setup_mode=:direct_near_null,
        nsmooth=0,
        verbose=false,
        coarse_vectors=[prototype],
        use_chirality=false,
        precompute_coarse=true,
        block_normalize=true)

    count = Ref(0)
    count_rhs = DDAMG._coarse_vector(coarse_setup)
    count_rhs.data .= 1
    DDAMG._amg_coarse_gmres_restart_boundary(
        coarse_setup, D, count_rhs;
        maxiter=1, restart=2, tol=1e-14,
        coarse_apply_counter=count)
    @test count[] == 4

    expected = DDAMG._copy_wf(prototype)
    r_expected = similar(template)
    DDAMG._amg_mul_dirac!(r_expected, D, expected)
    axpby!(1, expected, -1, r_expected)
    rc_expected = DDAMG.restrict_f2c(coarse_setup, r_expected)
    ec_expected = DDAMG._amg_coarse_gmres_restart_boundary(
        coarse_setup, D, rc_expected;
        maxiter=1, restart=2, tol=1e-2)
    correction_expected = DDAMG.prolongate_c2f(coarse_setup, ec_expected)
    DDAMG.sap_smooth!(correction_expected, D, r_expected, coarse_setup;
        niter=1, block_solver=:mr, nblock_mr=1,
        odd_even=false, sap_schedule=:twocolor)
    axpby!(1, correction_expected, 1, expected)
    DDAMG.normalize_fermion!(expected)

    actual = DDAMG._copy_wf(prototype)
    r_actual = similar(template)
    rc_actual = DDAMG._coarse_vector(coarse_setup)
    correction_actual = similar(template)
    DDAMG._amg_direct_near_null_update!(actual, D, coarse_setup,
        r_actual, rc_actual, correction_actual;
        update_smoother=:sap,
        post_sap_iters=1,
        block_solver=:mr,
        nblock_mr=1,
        nblock_gmres=1,
        coarse_tol=1e-2,
        coarse_maxiter=1,
        coarse_restart=2,
        sap_schedule=:twocolor,
        arnoldi_mode=:scalar)

    @test field_distance(actual, expected) < 1e-11
end

@testset "DDalphaAMG mode-specific solve defaults" begin
    @test DDAMG._amg_setup_defaults(:inverse_sap) == (
        vcycle_post_sap_iters=2,
        coarse_tol=5e-2,
        coarse_maxiter=5,
        coarse_restart=30,
        sap_schedule=:boundary8,
    )
    @test DDAMG._amg_setup_defaults(:direct_near_null) == (
        vcycle_post_sap_iters=1,
        coarse_tol=1e-2,
        coarse_maxiter=50,
        coarse_restart=20,
        sap_schedule=:twocolor,
    )
    @test DDAMG._amg_solve_defaults(:inverse_sap) == (
        restart=25,
        smoother=:sap,
        pre_smooth=0,
        post_smooth=2,
        coarse_tol=5e-2,
        coarse_maxiter=50,
        coarse_restart=30,
        coarse_odd_even=true,
        coarse_gmres_mode=:incremental,
        sap_odd_even=true,
        sap_schedule=:boundary8,
    )
    @test DDAMG._amg_solve_defaults(:direct_near_null) == (
        restart=50,
        smoother=:mr,
        pre_smooth=4,
        post_smooth=4,
        coarse_tol=1e-1,
        coarse_maxiter=50,
        coarse_restart=20,
        coarse_odd_even=false,
        coarse_gmres_mode=:restart_boundary,
        sap_odd_even=false,
        sap_schedule=:twocolor,
    )
end

@testset "DDalphaAMG solve options preserve explicit overrides" begin
    direct = DDAMG._amg_resolve_solve_options(:direct_near_null)
    @test direct == DDAMG._amg_solve_defaults(:direct_near_null)

    overridden = DDAMG._amg_resolve_solve_options(:direct_near_null;
        restart=25,
        coarse_odd_even=true,
        coarse_gmres_mode=:incremental)
    @test overridden.restart == 25
    @test overridden.coarse_odd_even
    @test overridden.smoother == :mr
    @test overridden.coarse_restart == 20
    @test overridden.coarse_gmres_mode == :incremental

    inverse = DDAMG._amg_resolve_solve_options(:inverse_sap)
    @test inverse == DDAMG._amg_solve_defaults(:inverse_sap)
end

@testset "DDalphaAMG direct solve uses restart-boundary coarse GMRES" begin
    template, D = make_setup_mode_cpu_problem()
    setup = DDAMG.setup_DDalphaAMG(
        D, template, 2, 1;
        setup_mode=:direct_near_null,
        nsmooth=0,
        nvcycle=0,
        use_chirality=false,
        precompute_coarse=false,
        verbose=false)
    rhs = similar(template)
    DDAMG.random_fermion!(rhs)
    counter = Ref(0)

    DDAMG.vcycle(
        D, rhs, setup;
        pre_smooth=0,
        post_smooth=0,
        smoother=:mr,
        coarse_maxiter=1,
        coarse_restart=2,
        coarse_tol=0.0,
        coarse_odd_even=false,
        coarse_gmres_mode=:restart_boundary,
        coarse_apply_counter=counter)

    @test counter[] == 4
end
