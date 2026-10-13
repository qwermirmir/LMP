# The Wilson Dirac operator on fermion fields: CPU operator aliases and the
# fused GPU kernel that applies D or D^dagger in one pass.

const WilsonDiracCPU =LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}

const AdjointWilsonDiracCPU =LatticeDiracOperators.Dirac_operators.Adjoint_Wilson_operator{LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}

const DdagDWilsonCPU =LatticeDiracOperators.Dirac_operators.DdagD_Wilson_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}

function Base.:*(D::WilsonDiracCPU,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end

function Base.:*(D::AdjointWilsonDiracCPU,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end

function Base.:*(x::WF3x1,D::WilsonDiracCPU)
    cx=conj(x)

    y=D'*cx

    return conj(y)
end 

function Base.:*(D::DdagDWilsonCPU,x::WF3x1)

    y=deepcopy(x)

    mul!(y,D,x)

    return y
end

# ============================================================
#  Fused Wilson Dirac kernel — replaces external mul!(y, D, x)
#  and mul!(y, D', x).  Zero allocations.
# ============================================================
#
# Convention (from LatticeDiracOperators):
#   D*x  = x - κ Σ_ν [ (r-γ_ν) U_ν(site) x(site+ν̂) +
#                        (r+γ_ν) U†_ν(site-ν̂) x(site-ν̂) ]
#   D†*x = x - κ Σ_ν [ (r+γ_ν) U_ν(site) x(site+ν̂) +
#                        (r-γ_ν) U†_ν(site-ν̂) x(site-ν̂) ]
#
# Gamma matrices (Euclidean, Dirac rep):
#   γ1: (1,4)=-i  (2,3)=-i  (3,2)=+i  (4,1)=+i
#   γ2: (1,4)=-1  (2,3)=+1  (3,2)=+1  (4,1)=-1
#   γ3: (1,3)=-i  (2,4)=+i  (3,1)=+i  (4,2)=-i
#   γ4: (1,3)=-1  (2,4)=-1  (3,1)=-1  (4,2)=-1
#
# With r=1, (1±γ_ν) has 2 nonzeros per row → spin projection.
# We hardcode these as inline functions for maximum performance.
# ============================================================

# ---- Spin projectors: each returns two 2-spinor components ----
# (1-γ_ν): used for forward hop in D, backward hop in D†
# (1+γ_ν): used for backward hop in D, forward hop in D†

# (1-γ₁)ψ:  row1 = ψ₁ + i·ψ₄,  row2 = ψ₂ + i·ψ₃,  row3 = ψ₃ - i·ψ₂,  row4 = ψ₄ - i·ψ₁
@inline _proj_m1(s1,s2,s3,s4) = (s1 + im*s4, s2 + im*s3, s3 - im*s2, s4 - im*s1)

# (1+γ₁)ψ:  row1 = ψ₁ - i·ψ₄,  row2 = ψ₂ - i·ψ₃,  row3 = ψ₃ + i·ψ₂,  row4 = ψ₄ + i·ψ₁
@inline _proj_p1(s1,s2,s3,s4) = (s1 - im*s4, s2 - im*s3, s3 + im*s2, s4 + im*s1)

# (1-γ₂)ψ:  row1 = ψ₁ + ψ₄,  row2 = ψ₂ - ψ₃,  row3 = ψ₃ - ψ₂,  row4 = ψ₄ + ψ₁
@inline _proj_m2(s1,s2,s3,s4) = (s1 + s4, s2 - s3, s3 - s2, s4 + s1)

# (1+γ₂)ψ:  row1 = ψ₁ - ψ₄,  row2 = ψ₂ + ψ₃,  row3 = ψ₃ + ψ₂,  row4 = ψ₄ - ψ₁
@inline _proj_p2(s1,s2,s3,s4) = (s1 - s4, s2 + s3, s3 + s2, s4 - s1)

# (1-γ₃)ψ:  row1 = ψ₁ + i·ψ₃,  row2 = ψ₂ - i·ψ₄,  row3 = ψ₃ - i·ψ₁,  row4 = ψ₄ + i·ψ₂
@inline _proj_m3(s1,s2,s3,s4) = (s1 + im*s3, s2 - im*s4, s3 - im*s1, s4 + im*s2)

# (1+γ₃)ψ:  row1 = ψ₁ - i·ψ₃,  row2 = ψ₂ + i·ψ₄,  row3 = ψ₃ + i·ψ₁,  row4 = ψ₄ - i·ψ₂
@inline _proj_p3(s1,s2,s3,s4) = (s1 - im*s3, s2 + im*s4, s3 + im*s1, s4 - im*s2)

# (1-γ₄)ψ:  row1 = ψ₁ + ψ₃,  row2 = ψ₂ + ψ₄,  row3 = ψ₃ + ψ₁,  row4 = ψ₄ + ψ₂
@inline _proj_m4(s1,s2,s3,s4) = (s1 + s3, s2 + s4, s3 + s1, s4 + s2)

# (1+γ₄)ψ:  row1 = ψ₁ - ψ₃,  row2 = ψ₂ - ψ₄,  row3 = ψ₃ - ψ₁,  row4 = ψ₄ - ψ₂
@inline _proj_p4(s1,s2,s3,s4) = (s1 - s3, s2 - s4, s3 - s1, s4 - s2)

function _cudakernel_fused_wilson!(
    yf     ::CuDeviceArray{ComplexF64, 4},   # output (3, 4, blocksize, rsize)
    xf     ::CuDeviceArray{ComplexF64, 4},   # input
    U1     ::CuDeviceArray{ComplexF64, 4},   # gauge μ=1 (3,3,bs,rs)
    U2     ::CuDeviceArray{ComplexF64, 4},
    U3     ::CuDeviceArray{ComplexF64, 4},
    U4     ::CuDeviceArray{ComplexF64, 4},
    κ      ::Float64,
    blockinfo,
    bc,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    is_dagger::Bool   # false = D, true = D†
)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    # Read x at current site
    @inbounds begin
        x1_1 = xf[1,1,b,r]; x2_1 = xf[2,1,b,r]; x3_1 = xf[3,1,b,r]
        x1_2 = xf[1,2,b,r]; x2_2 = xf[2,2,b,r]; x3_2 = xf[3,2,b,r]
        x1_3 = xf[1,3,b,r]; x2_3 = xf[2,3,b,r]; x3_3 = xf[3,3,b,r]
        x1_4 = xf[1,4,b,r]; x2_4 = xf[2,4,b,r]; x3_4 = xf[3,4,b,r]
    end

    # Accumulators for temp = Σ_ν contribution (per color, per spin)
    a1_1=zero(ComplexF64); a2_1=zero(ComplexF64); a3_1=zero(ComplexF64)
    a1_2=zero(ComplexF64); a2_2=zero(ComplexF64); a3_2=zero(ComplexF64)
    a1_3=zero(ComplexF64); a2_3=zero(ComplexF64); a3_3=zero(ComplexF64)
    a1_4=zero(ComplexF64); a2_4=zero(ComplexF64); a3_4=zero(ComplexF64)

    # ---- Process each direction ν = 1..4 ----
    for ν in 1:4
        Uν = ν == 1 ? U1 : (ν == 2 ? U2 : (ν == 3 ? U3 : U4))

        # ------ Forward hop: xplus = x(site+ν̂), temp1 = U_ν(site) * xplus ------
        shift_fwd = (Int8(ν==1), Int8(ν==2), Int8(ν==3), Int8(ν==4))
        bs_f, rs_f = _shifted_idx(b, r, shift_fwd, blockinfo)
        fac_f = _bc_factor(b, r, shift_fwd, blockinfo, bc, NX, NY, NZ, NT)

        @inbounds begin
            # Read shifted fermion x(site+ν̂)
            xp1_1 = xf[1,1,bs_f,rs_f]*fac_f; xp2_1 = xf[2,1,bs_f,rs_f]*fac_f; xp3_1 = xf[3,1,bs_f,rs_f]*fac_f
            xp1_2 = xf[1,2,bs_f,rs_f]*fac_f; xp2_2 = xf[2,2,bs_f,rs_f]*fac_f; xp3_2 = xf[3,2,bs_f,rs_f]*fac_f
            xp1_3 = xf[1,3,bs_f,rs_f]*fac_f; xp2_3 = xf[2,3,bs_f,rs_f]*fac_f; xp3_3 = xf[3,3,bs_f,rs_f]*fac_f
            xp1_4 = xf[1,4,bs_f,rs_f]*fac_f; xp2_4 = xf[2,4,bs_f,rs_f]*fac_f; xp3_4 = xf[3,4,bs_f,rs_f]*fac_f

            # U_ν(b,r) * xplus — for each spin component
            u11=Uν[1,1,b,r]; u12=Uν[1,2,b,r]; u13=Uν[1,3,b,r]
            u21=Uν[2,1,b,r]; u22=Uν[2,2,b,r]; u23=Uν[2,3,b,r]
            u31=Uν[3,1,b,r]; u32=Uν[3,2,b,r]; u33=Uν[3,3,b,r]
        end
        # temp1[color, spin] = U * xplus  (per spin)
        t1_1 = u11*xp1_1 + u12*xp2_1 + u13*xp3_1
        t2_1 = u21*xp1_1 + u22*xp2_1 + u23*xp3_1
        t3_1 = u31*xp1_1 + u32*xp2_1 + u33*xp3_1
        t1_2 = u11*xp1_2 + u12*xp2_2 + u13*xp3_2
        t2_2 = u21*xp1_2 + u22*xp2_2 + u23*xp3_2
        t3_2 = u31*xp1_2 + u32*xp2_2 + u33*xp3_2
        t1_3 = u11*xp1_3 + u12*xp2_3 + u13*xp3_3
        t2_3 = u21*xp1_3 + u22*xp2_3 + u23*xp3_3
        t3_3 = u31*xp1_3 + u32*xp2_3 + u33*xp3_3
        t1_4 = u11*xp1_4 + u12*xp2_4 + u13*xp3_4
        t2_4 = u21*xp1_4 + u22*xp2_4 + u23*xp3_4
        t3_4 = u31*xp1_4 + u32*xp2_4 + u33*xp3_4

        # Apply spin projector to temp1: D uses (r-γ_ν), D† uses (r+γ_ν)
        # Apply per color independently (spin projector acts on Dirac indices only)
        if !is_dagger
            # Forward hop in D: (1-γ_ν)
            if ν == 1
                p1_1,p1_2,p1_3,p1_4 = _proj_m1(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m1(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m1(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 2
                p1_1,p1_2,p1_3,p1_4 = _proj_m2(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m2(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m2(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 3
                p1_1,p1_2,p1_3,p1_4 = _proj_m3(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m3(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m3(t3_1,t3_2,t3_3,t3_4)
            else
                p1_1,p1_2,p1_3,p1_4 = _proj_m4(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_m4(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_m4(t3_1,t3_2,t3_3,t3_4)
            end
        else
            # Forward hop in D†: (1+γ_ν)
            if ν == 1
                p1_1,p1_2,p1_3,p1_4 = _proj_p1(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p1(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p1(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 2
                p1_1,p1_2,p1_3,p1_4 = _proj_p2(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p2(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p2(t3_1,t3_2,t3_3,t3_4)
            elseif ν == 3
                p1_1,p1_2,p1_3,p1_4 = _proj_p3(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p3(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p3(t3_1,t3_2,t3_3,t3_4)
            else
                p1_1,p1_2,p1_3,p1_4 = _proj_p4(t1_1,t1_2,t1_3,t1_4)
                p2_1,p2_2,p2_3,p2_4 = _proj_p4(t2_1,t2_2,t2_3,t2_4)
                p3_1,p3_2,p3_3,p3_4 = _proj_p4(t3_1,t3_2,t3_3,t3_4)
            end
        end

        # Accumulate forward contribution (κ * projected)
        a1_1 += κ*p1_1; a2_1 += κ*p2_1; a3_1 += κ*p3_1
        a1_2 += κ*p1_2; a2_2 += κ*p2_2; a3_2 += κ*p3_2
        a1_3 += κ*p1_3; a2_3 += κ*p2_3; a3_3 += κ*p3_3
        a1_4 += κ*p1_4; a2_4 += κ*p2_4; a3_4 += κ*p3_4

        # ------ Backward hop: xminus = x(site-ν̂), temp2 = U†_ν(site-ν̂) * xminus ------
        shift_bwd = (-Int8(ν==1), -Int8(ν==2), -Int8(ν==3), -Int8(ν==4))
        bs_b, rs_b = _shifted_idx(b, r, shift_bwd, blockinfo)
        fac_b = _bc_factor(b, r, shift_bwd, blockinfo, bc, NX, NY, NZ, NT)

        @inbounds begin
            xm1_1 = xf[1,1,bs_b,rs_b]*fac_b; xm2_1 = xf[2,1,bs_b,rs_b]*fac_b; xm3_1 = xf[3,1,bs_b,rs_b]*fac_b
            xm1_2 = xf[1,2,bs_b,rs_b]*fac_b; xm2_2 = xf[2,2,bs_b,rs_b]*fac_b; xm3_2 = xf[3,2,bs_b,rs_b]*fac_b
            xm1_3 = xf[1,3,bs_b,rs_b]*fac_b; xm2_3 = xf[2,3,bs_b,rs_b]*fac_b; xm3_3 = xf[3,3,bs_b,rs_b]*fac_b
            xm1_4 = xf[1,4,bs_b,rs_b]*fac_b; xm2_4 = xf[2,4,bs_b,rs_b]*fac_b; xm3_4 = xf[3,4,bs_b,rs_b]*fac_b

            # U†_ν(site-ν̂): conj transpose of U_ν at shifted site
            v11=conj(Uν[1,1,bs_b,rs_b]); v12=conj(Uν[2,1,bs_b,rs_b]); v13=conj(Uν[3,1,bs_b,rs_b])
            v21=conj(Uν[1,2,bs_b,rs_b]); v22=conj(Uν[2,2,bs_b,rs_b]); v23=conj(Uν[3,2,bs_b,rs_b])
            v31=conj(Uν[1,3,bs_b,rs_b]); v32=conj(Uν[2,3,bs_b,rs_b]); v33=conj(Uν[3,3,bs_b,rs_b])
        end
        # temp2[color,spin] = U† * xminus
        q1_1 = v11*xm1_1 + v12*xm2_1 + v13*xm3_1
        q2_1 = v21*xm1_1 + v22*xm2_1 + v23*xm3_1
        q3_1 = v31*xm1_1 + v32*xm2_1 + v33*xm3_1
        q1_2 = v11*xm1_2 + v12*xm2_2 + v13*xm3_2
        q2_2 = v21*xm1_2 + v22*xm2_2 + v23*xm3_2
        q3_2 = v31*xm1_2 + v32*xm2_2 + v33*xm3_2
        q1_3 = v11*xm1_3 + v12*xm2_3 + v13*xm3_3
        q2_3 = v21*xm1_3 + v22*xm2_3 + v23*xm3_3
        q3_3 = v31*xm1_3 + v32*xm2_3 + v33*xm3_3
        q1_4 = v11*xm1_4 + v12*xm2_4 + v13*xm3_4
        q2_4 = v21*xm1_4 + v22*xm2_4 + v23*xm3_4
        q3_4 = v31*xm1_4 + v32*xm2_4 + v33*xm3_4

        # Apply spin projector to temp2: D uses (r+γ_ν), D† uses (r-γ_ν)
        if !is_dagger
            if ν == 1
                r1_1,r1_2,r1_3,r1_4 = _proj_p1(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p1(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p1(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 2
                r1_1,r1_2,r1_3,r1_4 = _proj_p2(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p2(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p2(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 3
                r1_1,r1_2,r1_3,r1_4 = _proj_p3(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p3(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p3(q3_1,q3_2,q3_3,q3_4)
            else
                r1_1,r1_2,r1_3,r1_4 = _proj_p4(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_p4(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_p4(q3_1,q3_2,q3_3,q3_4)
            end
        else
            if ν == 1
                r1_1,r1_2,r1_3,r1_4 = _proj_m1(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m1(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m1(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 2
                r1_1,r1_2,r1_3,r1_4 = _proj_m2(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m2(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m2(q3_1,q3_2,q3_3,q3_4)
            elseif ν == 3
                r1_1,r1_2,r1_3,r1_4 = _proj_m3(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m3(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m3(q3_1,q3_2,q3_3,q3_4)
            else
                r1_1,r1_2,r1_3,r1_4 = _proj_m4(q1_1,q1_2,q1_3,q1_4)
                r2_1,r2_2,r2_3,r2_4 = _proj_m4(q2_1,q2_2,q2_3,q2_4)
                r3_1,r3_2,r3_3,r3_4 = _proj_m4(q3_1,q3_2,q3_3,q3_4)
            end
        end

        # Accumulate backward contribution
        a1_1 += κ*r1_1; a2_1 += κ*r2_1; a3_1 += κ*r3_1
        a1_2 += κ*r1_2; a2_2 += κ*r2_2; a3_2 += κ*r3_2
        a1_3 += κ*r1_3; a2_3 += κ*r2_3; a3_3 += κ*r3_3
        a1_4 += κ*r1_4; a2_4 += κ*r2_4; a3_4 += κ*r3_4
    end  # ν loop

    # y = x - accumulated
    @inbounds begin
        yf[1,1,b,r] = x1_1 - a1_1; yf[2,1,b,r] = x2_1 - a2_1; yf[3,1,b,r] = x3_1 - a3_1
        yf[1,2,b,r] = x1_2 - a1_2; yf[2,2,b,r] = x2_2 - a2_2; yf[3,2,b,r] = x3_2 - a3_2
        yf[1,3,b,r] = x1_3 - a1_3; yf[2,3,b,r] = x2_3 - a2_3; yf[3,3,b,r] = x3_3 - a3_3
        yf[1,4,b,r] = x1_4 - a1_4; yf[2,4,b,r] = x2_4 - a2_4; yf[3,4,b,r] = x3_4 - a3_4
    end
    return nothing
end

"""
    fused_wilson_mul_gpu!(y, x, U, κ, is_dagger)

Fused Wilson Dirac operator on GPU.  Zero allocations.
  - is_dagger=false: y = D*x
  - is_dagger=true:  y = D†*x
Replaces external LatticeDiracOperators mul!(y, D, x) which allocates
shift_fermion temporaries (~40 KB per call).
"""
function fused_wilson_mul_gpu!(y::WF3x1GPU, x::WF3x1GPU,
                               U::GPUgauge, κ::Float64, is_dagger::Bool)
    bi = x.blockinfo
    blocksize = bi.blocksize
    rsize     = bi.rsize
    bc = _BC_DEFAULT
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_wilson!(
        y.f, x.f, U[1].U, U[2].U, U[3].U, U[4].U,
        κ, bi, bc, NX, NY, NZ, NT, is_dagger)
end

@inline _dirac_kappa(D) = getproperty(D, Symbol("\u03ba"))
