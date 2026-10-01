# Parallel transport of fermion fields along paths (Hopping_operator), on the
# CPU and the GPU, and the batched GPU kernels that transport many paths at
# once.

"""
    Hopping_operator(U, P)

Parallel transport of fermion fields along the paths `P` through the links
`U`. `hop(x)` returns one transported copy of `x` per path, and `hop(x, V)`
writes them into `V`. A path is applied step by step:

    step +mu:  (T psi)(n) = U_mu(n - mu)^dagger psi(n - mu)
    step -mu:  (T psi)(n) = U_mu(n) psi(n + mu)
    step 0:    no transport

so the field moves one site along +mu or -mu and stays gauge covariant. The
fermion boundary condition is periodic in space and antiperiodic in time.
"""
mutable struct Hopping_operator{G<:AbstractGaugefields}
    Gauge::Vector{G}
    P::Paths
end

Base.:length(hop::Hopping_operator) = length(hop.P)

# ---------------------------------------------------------------------------
# CPU transport
# ---------------------------------------------------------------------------

function (hop::Hopping_operator)(x::WF3x1CPU)

    U = hop.Gauge
    shifts = Vector{WF3x1}()

    for k in 1:length(hop.P.various)
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)

                mul!(copyx, Uminus',shift_x)


            elseif a < 0

                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)
            
            elseif a == 0

                copyx = deepcopy(x)
                
            end
        end

        push!(shifts, copyx)

    end

    return shifts
end

function (hop::Hopping_operator)(x::Vector{WF3x1CPU})

    U = hop.Gauge
    shifts = Vector{WF3x1}()

    for k in 1:length(hop.P.various)
        copyx = deepcopy(x[k])
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)

                mul!(copyx, Uminus',shift_x)
            
            elseif a < 0
                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)
            elseif a == 0
                copyx = deepcopy(x[k])
                
            end
        end

        push!(shifts, copyx)

    end
    return shifts
end

# Writes the transported copies of x into V, one per path, without allocating.
function (hop::Hopping_operator)(x::WF3x1CPU, V::AbstractVector{T}) where {T<:WF3x1CPU}
    @assert length(V) == length(hop.P.various) "length(V) must equal the number of paths."
    
    U = hop.Gauge

    @inbounds for k in eachindex(V)
        copyx = V[k]

        _copy_fermion!(copyx, x)

        for step in hop.P.various[k].steps
            a = step.f
            if a > 0
                _hop_step_forward!(copyx, U[a], a)

            elseif a < 0
                _hop_step_backward!(copyx, U[-a], -a)
            
            else
                _copy_fermion!(copyx, x)
            end
        end
    end

    return V
end

# dest .= src on the CPU, without allocating.
function _copy_fermion!(dest::WF3x1CPU, src::WF3x1CPU)
    @assert size(dest.f) == size(src.f) "dest and src must have the same size."
    copyto!(dest.f, src.f)
    return dest
end

# Single transport steps that write into the field's own buffers instead of
# allocating shifted copies.

const _bc_default = Int8[1, 1, 1, -1]

# F.fshifted(n) = bc * F.f(n + shift), with the boundary sign bc.
function _shifted_fermion_inplace!(F::WF3x1CPU, shift::NTuple{4,Int})
    bc = _bc_default
    NX = F.NX; NY = F.NY; NZ = F.NZ; NT = F.NT
    sf = F.f; fsh = F.fshifted

    @inbounds for ig = 1:4
        for it = 1:NT
            it_s = it + shift[4]
            inside_up = it_s > NT; inside_down = it_s < 1
            factor_t = ifelse(inside_up || inside_down, Int(bc[4]), 1)
            it_s += ifelse(inside_up, -NT, 0)
            it_s += ifelse(inside_down, NT, 0)
            for iz = 1:NZ
                iz_s = iz + shift[3]
                inside_up = iz_s > NZ; inside_down = iz_s < 1
                factor_z = ifelse(inside_up || inside_down, Int(bc[3]), 1)
                iz_s += ifelse(inside_up, -NZ, 0)
                iz_s += ifelse(inside_down, NZ, 0)
                for iy = 1:NY
                    iy_s = iy + shift[2]
                    inside_up = iy_s > NY; inside_down = iy_s < 1
                    factor_y = ifelse(inside_up || inside_down, Int(bc[2]), 1)
                    iy_s += ifelse(inside_up, -NY, 0)
                    iy_s += ifelse(inside_down, NY, 0)
                    for ix = 1:NX
                        ix_s = ix + shift[1]
                        inside_up = ix_s > NX; inside_down = ix_s < 1
                        factor_x = ifelse(inside_up || inside_down, Int(bc[1]), 1)
                        ix_s += ifelse(inside_up, -NX, 0)
                        ix_s += ifelse(inside_down, NX, 0)
                        factor = factor_x * factor_y * factor_z * factor_t
                        @simd for ic = 1:3
                            fsh[ic, ix, iy, iz, it, ig] = factor * sf[ic, ix_s, iy_s, iz_s, it_s, ig]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# Step +mu: copyx(n) <- U_mu(n - mu)^dagger copyx(n - mu). The link is read at
# the shifted site directly, so no shifted copy of U is made.
function _hop_step_forward!(copyx::WF3x1CPU, U_mu, mu::Int)
    bc = _bc_default
    NX = copyx.NX; NY = copyx.NY; NZ = copyx.NZ; NT = copyx.NT
    sf = copyx.f; fsh = copyx.fshifted
    Ug = U_mu.U

    shift1 = (mu == 1) ? -1 : 0
    shift2 = (mu == 2) ? -1 : 0
    shift3 = (mu == 3) ? -1 : 0
    shift4 = (mu == 4) ? -1 : 0

    @inbounds for ig = 1:4
        for it = 1:NT
            it_s = it + shift4
            inside_up = it_s > NT; inside_down = it_s < 1
            factor_t = ifelse(inside_up || inside_down, Int(bc[4]), 1)
            it_s += ifelse(inside_up, -NT, 0)
            it_s += ifelse(inside_down, NT, 0)
            for iz = 1:NZ
                iz_s = iz + shift3
                inside_up = iz_s > NZ; inside_down = iz_s < 1
                factor_z = ifelse(inside_up || inside_down, Int(bc[3]), 1)
                iz_s += ifelse(inside_up, -NZ, 0)
                iz_s += ifelse(inside_down, NZ, 0)
                for iy = 1:NY
                    iy_s = iy + shift2
                    inside_up = iy_s > NY; inside_down = iy_s < 1
                    factor_y = ifelse(inside_up || inside_down, Int(bc[2]), 1)
                    iy_s += ifelse(inside_up, -NY, 0)
                    iy_s += ifelse(inside_down, NY, 0)
                    for ix = 1:NX
                        ix_s = ix + shift1
                        inside_up = ix_s > NX; inside_down = ix_s < 1
                        factor_x = ifelse(inside_up || inside_down, Int(bc[1]), 1)
                        ix_s += ifelse(inside_up, -NX, 0)
                        ix_s += ifelse(inside_down, NX, 0)

                        factor = factor_x * factor_y * factor_z * factor_t

                        # Read shifted fermion
                        x1 = factor * sf[1, ix_s, iy_s, iz_s, it_s, ig]
                        x2 = factor * sf[2, ix_s, iy_s, iz_s, it_s, ig]
                        x3 = factor * sf[3, ix_s, iy_s, iz_s, it_s, ig]

                        # Read gauge at shifted index (no shift_U needed)
                        # Compute U†*x: (U†)[i,j] = conj(U[j,i])
                        fsh[1, ix, iy, iz, it, ig] =
                            conj(Ug[1,1,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,1,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,1,ix_s,iy_s,iz_s,it_s])*x3
                        fsh[2, ix, iy, iz, it, ig] =
                            conj(Ug[1,2,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,2,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,2,ix_s,iy_s,iz_s,it_s])*x3
                        fsh[3, ix, iy, iz, it, ig] =
                            conj(Ug[1,3,ix_s,iy_s,iz_s,it_s])*x1 +
                            conj(Ug[2,3,ix_s,iy_s,iz_s,it_s])*x2 +
                            conj(Ug[3,3,ix_s,iy_s,iz_s,it_s])*x3
                    end
                end
            end
        end
    end
    copyto!(sf, fsh)
    return nothing
end

# Step -mu: copyx(n) <- U_mu(n) copyx(n + mu).
function _hop_step_backward!(copyx::WF3x1CPU, U_mu, mu::Int)
    shift = ntuple(i -> (i == mu) ? 1 : 0, 4)
    _shifted_fermion_inplace!(copyx, shift)

    fsh = copyx.fshifted
    Ug = U_mu.U
    df = copyx.f
    NX = copyx.NX; NY = copyx.NY; NZ = copyx.NZ; NT = copyx.NT

    @inbounds for ig = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        x1 = fsh[1, ix, iy, iz, it, ig]
                        x2 = fsh[2, ix, iy, iz, it, ig]
                        x3 = fsh[3, ix, iy, iz, it, ig]

                        df[1, ix, iy, iz, it, ig] =
                            Ug[1,1,ix,iy,iz,it]*x1 + Ug[1,2,ix,iy,iz,it]*x2 + Ug[1,3,ix,iy,iz,it]*x3
                        df[2, ix, iy, iz, it, ig] =
                            Ug[2,1,ix,iy,iz,it]*x1 + Ug[2,2,ix,iy,iz,it]*x2 + Ug[2,3,ix,iy,iz,it]*x3
                        df[3, ix, iy, iz, it, ig] =
                            Ug[3,1,ix,iy,iz,it]*x1 + Ug[3,2,ix,iy,iz,it]*x2 + Ug[3,3,ix,iy,iz,it]*x3
                    end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# GPU transport
# ---------------------------------------------------------------------------

function (hop::Hopping_operator)(x::WF3x1GPU)

    shifts = Vector{WF3x1GPU}()

    for k in 1:length(hop.P.various)
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                mul!(copyx,hop.Gauge'[a],copyx)
                shift_x=shift_fermion(copyx , -a)
                substitute_fermion!(copyx,shift_x)

            elseif a < 0
                a = -a
                shift_x=shift_fermion(copyx, a)
                mul!(copyx, hop.Gauge[a], shift_x)
            elseif a == 0
                copyx = deepcopy(x)     
            end
        end
        push!(shifts,copyx)
    end

    return shifts
end

function (hop::Hopping_operator)( x::WF3x1GPU,V::AbstractVector{T}) where {T<:WF3x1GPU}
    @assert length(V) == length(hop.P.various)

    @inbounds for k in eachindex(V)
        copyx = V[k]

        substitute_fermion!(copyx, x)

        for step in hop.P.various[k].steps
            a = step.f

            if a > 0
                mul!(copyx, adjoint(hop.Gauge[a]), copyx)

                # shift_fermion allocates a shifted copy.
                shift_x = shift_fermion(copyx, -a)
                substitute_fermion!(copyx, shift_x)

            elseif a < 0
                aa = -a
                shift_x = shift_fermion(copyx, aa)
                mul!(copyx, hop.Gauge[aa], shift_x)

            else
                substitute_fermion!(copyx, x)
            end
        end
    end

    return V
end

# ---------------------------------------------------------------------------
# Batched GPU transport
# ---------------------------------------------------------------------------

# Transports the fields of all paths together, one kernel launch per step
# instead of one per path and step. Paths are padded to max_steps; a path
# that has finished, or an empty step, copies its field through.

function _cudakernel_batched_hop_step!(
    dst, src, U1, U2, U3, U4,
    dirs,         # CuDeviceArray{Int8,2} (max_steps, npaths)
    nsteps_arr,   # CuDeviceArray{Int32,1} (npaths,)
    step_idx,     # Int32
    npaths,       # Int32
    blockinfo, bc,
    NX::Int, NY::Int, NZ::Int, NT::Int
)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for pidx in Int32(1):npaths
        if step_idx > nsteps_arr[pidx]
            # Copy-through: path already done
            for g in 1:4
                dst[1,g,b,r,pidx] = src[1,g,b,r,pidx]
                dst[2,g,b,r,pidx] = src[2,g,b,r,pidx]
                dst[3,g,b,r,pidx] = src[3,g,b,r,pidx]
            end
        else
            d = Int(dirs[step_idx, pidx])
            if d > 0
                # positive direction: y(b,r) = U†_d(bs,rs) * x(bs,rs) * fac
                s1 = Int8(d==1 ? -1 : 0); s2 = Int8(d==2 ? -1 : 0)
                s3 = Int8(d==3 ? -1 : 0); s4 = Int8(d==4 ? -1 : 0)
                shift = (s1, s2, s3, s4)
                bs, rs = _shifted_idx(b, r, shift, blockinfo)
                fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
                Uν = d == 1 ? U1 : (d == 2 ? U2 : (d == 3 ? U3 : U4))
                u11=Uν[1,1,bs,rs]; u21=Uν[2,1,bs,rs]; u31=Uν[3,1,bs,rs]
                u12=Uν[1,2,bs,rs]; u22=Uν[2,2,bs,rs]; u32=Uν[3,2,bs,rs]
                u13=Uν[1,3,bs,rs]; u23=Uν[2,3,bs,rs]; u33=Uν[3,3,bs,rs]
                for alpha in 1:4
                    x1 = src[1,alpha,bs,rs,pidx]*fac
                    x2 = src[2,alpha,bs,rs,pidx]*fac
                    x3 = src[3,alpha,bs,rs,pidx]*fac
                    dst[1,alpha,b,r,pidx] = conj(u11)*x1 + conj(u21)*x2 + conj(u31)*x3
                    dst[2,alpha,b,r,pidx] = conj(u12)*x1 + conj(u22)*x2 + conj(u32)*x3
                    dst[3,alpha,b,r,pidx] = conj(u13)*x1 + conj(u23)*x2 + conj(u33)*x3
                end
            elseif d < 0
                # negative direction: y(b,r) = U_|d|(b,r) * x(bs,rs) * fac
                aa = -d
                s1 = Int8(aa==1 ? 1 : 0); s2 = Int8(aa==2 ? 1 : 0)
                s3 = Int8(aa==3 ? 1 : 0); s4 = Int8(aa==4 ? 1 : 0)
                shift = (s1, s2, s3, s4)
                bs, rs = _shifted_idx(b, r, shift, blockinfo)
                fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
                Uν = aa == 1 ? U1 : (aa == 2 ? U2 : (aa == 3 ? U3 : U4))
                u11=Uν[1,1,b,r]; u12=Uν[1,2,b,r]; u13=Uν[1,3,b,r]
                u21=Uν[2,1,b,r]; u22=Uν[2,2,b,r]; u23=Uν[2,3,b,r]
                u31=Uν[3,1,b,r]; u32=Uν[3,2,b,r]; u33=Uν[3,3,b,r]
                for alpha in 1:4
                    x1 = src[1,alpha,bs,rs,pidx]*fac
                    x2 = src[2,alpha,bs,rs,pidx]*fac
                    x3 = src[3,alpha,bs,rs,pidx]*fac
                    dst[1,alpha,b,r,pidx] = u11*x1 + u12*x2 + u13*x3
                    dst[2,alpha,b,r,pidx] = u21*x1 + u22*x2 + u23*x3
                    dst[3,alpha,b,r,pidx] = u31*x1 + u32*x2 + u33*x3
                end
            else
                # d == 0: identity step — copy through
                for g in 1:4
                    dst[1,g,b,r,pidx] = src[1,g,b,r,pidx]
                    dst[2,g,b,r,pidx] = src[2,g,b,r,pidx]
                    dst[3,g,b,r,pidx] = src[3,g,b,r,pidx]
                end
            end
        end
    end
    return nothing
end

"""
    batched_hopping_steps_gpu!(hopped_all, temp_5d, x_fields, Gauge,
                                dirs_gpu, nsteps_gpu, max_steps, npaths,
                                blocksize, rsize)

Packs `x_fields` into `hopped_all` and transports field i along path i, one
kernel per step, alternating between `hopped_all` and `temp_5d`. The result
is left in `hopped_all`.
"""
function batched_hopping_steps_gpu!(hopped_all, temp_5d, x_fields,
                                    Gauge, dirs_gpu, nsteps_gpu, max_steps, npaths,
                                    blocksize, rsize)
    bi = x_fields[1].blockinfo
    bc = _BC_DEFAULT
    NX = x_fields[1].NX; NY = x_fields[1].NY
    NZ = x_fields[1].NZ; NT = x_fields[1].NT

    # 1) Pack all x[i] → hopped_all
    pack_fermions!(hopped_all, x_fields)

    # 2) Step-by-step batched hop with double-buffering
    buf_a = hopped_all
    buf_b = temp_5d
    for s in Int32(1):Int32(max_steps)
        CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_hop_step!(
            buf_b, buf_a,
            Gauge[1].U, Gauge[2].U, Gauge[3].U, Gauge[4].U,
            dirs_gpu, nsteps_gpu, s, Int32(npaths),
            bi, bc, NX, NY, NZ, NT)
        buf_a, buf_b = buf_b, buf_a
    end

    # 3) After max_steps swaps: if even, buf_a=hopped_all; if odd, buf_a=temp_5d
    if isodd(max_steps)
        copyto!(hopped_all, temp_5d)
    end
end

"""
    batched_inv_hopping_gpu!(ci_all, temp_5d, Gauge,
                              inv_dirs_gpu, nsteps_gpu, max_steps, npaths,
                              blocksize, rsize, bi, NX, NY, NZ, NT)

Transports the packed fields in `ci_all` along the reversed paths for the
PTLayer backward pass, alternating between `ci_all` and `temp_5d`. The result
is left in `ci_all`.
"""
function batched_inv_hopping_gpu!(ci_all, temp_5d, Gauge,
                                   inv_dirs_gpu, nsteps_gpu, max_steps, npaths,
                                   blocksize, rsize, bi, NX, NY, NZ, NT)
    bc = _BC_DEFAULT

    buf_a = ci_all
    buf_b = temp_5d
    for s in Int32(1):Int32(max_steps)
        CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_batched_hop_step!(
            buf_b, buf_a,
            Gauge[1].U, Gauge[2].U, Gauge[3].U, Gauge[4].U,
            inv_dirs_gpu, nsteps_gpu, s, Int32(npaths),
            bi, bc, NX, NY, NZ, NT)
        buf_a, buf_b = buf_b, buf_a
    end

    if isodd(max_steps)
        copyto!(ci_all, temp_5d)
    end
end
