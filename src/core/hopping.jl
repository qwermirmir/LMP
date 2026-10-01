# Parallel transport of fermion fields along paths (Hopping_operator), on the
# CPU and on the GPU, including the batched GPU kernels used by PTLayer.

# CPU 버전 _copy_fermion! - inplace copy (메모리 할당 없음)
function _copy_fermion!(dest::WF3x1CPU, src::WF3x1CPU)
    @assert size(dest.f) == size(src.f) "dest와 src의 크기가 같아야 합니다."
    copyto!(dest.f, src.f)
    return dest
end

mutable struct Hopping_operator{G<:AbstractGaugefields}
    Gauge::Vector{G}
    P::Paths
end

function (hop::Hopping_operator)(x::WF3x1CPU)

    U = hop.Gauge
    shifts = Vector{WF3x1}()

    for k in 1:length(hop.P.various)
        # Ucopy 삭제 - U는 읽기만 하므로 deepcopy 불필요
        copyx = deepcopy(x)
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)  # Ucopy 대신 U 사용

                mul!(copyx, Uminus',shift_x)


            elseif a < 0

                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)  # Ucopy 대신 U 사용
            
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
        # Ucopy 삭제 - U는 읽기만 하므로 deepcopy 불필요
        copyx = deepcopy(x[k])
        for step in hop.P.various[k].steps
            a = step.f
            if a > 0

                shift_x=shift_fermion(copyx , -a)
                Uminus = shift_U(U[a], -a)  # Ucopy 대신 U 사용

                mul!(copyx, Uminus',shift_x)
            
            elseif a < 0
                a = -a

                shift_x=shift_fermion(copyx, a)

                mul!(copyx, U[a], shift_x)  # Ucopy 대신 U 사용
            elseif a == 0
                copyx = deepcopy(x[k])
                
            end
        end

        push!(shifts, copyx)

    end
    return shifts
end

# ============================================================
# Zero-allocation hop step 함수들
# shift_fermion/shift_U의 래퍼 구조체 할당을 완전히 제거
# ============================================================

const _bc_default = Int8[1, 1, 1, -1]

# shifted_fermion! 인라인 버전: copyx.f → copyx.fshifted (BC 적용)
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

# Forward step: copyx[site] = U†(site-μ̂) * copyx_old(site-μ̂)
# shift_U 제거: 게이지도 같은 shifted 인덱스에서 직접 읽음
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
    # fshifted → f 복사
    copyto!(sf, fsh)
    return nothing
end

# Backward step: copyx[site] = U_μ(site) * copyx_old(site+μ̂)
# 게이지는 shift 불필요 (현재 site에서 읽음)
function _hop_step_backward!(copyx::WF3x1CPU, U_mu, mu::Int)
    # fermion만 shift: copyx.f → copyx.fshifted
    shift = ntuple(i -> (i == mu) ? 1 : 0, 4)
    _shifted_fermion_inplace!(copyx, shift)

    # fshifted에서 읽고 U*fshifted 결과를 f에 씀
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

# Preallocated 버퍼를 받는 버전 - 메모리 할당 완전 제거
function (hop::Hopping_operator)(x::WF3x1CPU, V::AbstractVector{T}) where {T<:WF3x1CPU}
    @assert length(V) == length(hop.P.various) "V의 길이는 paths의 개수와 같아야 합니다."
    
    U = hop.Gauge

    @inbounds for k in eachindex(V)
        copyx = V[k]
        
        # copyx를 x로 초기화 (inplace copy)
        _copy_fermion!(copyx, x)

        for step in hop.P.various[k].steps
            a = step.f
            if a > 0
                _hop_step_forward!(copyx, U[a], a)

            elseif a < 0
                _hop_step_backward!(copyx, U[-a], -a)
            
            else
                # a == 0: copyx를 x로 재설정
                _copy_fermion!(copyx, x)
            end
        end
    end

    return V
end

function Base.:length(hop::Hopping_operator)
    return length(hop.P)

end

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

        # copyx를 "x로 초기화" (새로 할당 X)
        substitute_fermion!(copyx, x)   # 당신 코드에 이미 있는 in-place copy라고 가정

        for step in hop.P.various[k].steps
            a = step.f

            if a > 0
                mul!(copyx, adjoint(hop.Gauge[a]), copyx)

                # 여기 shift_fermion이 out-of-place면 여기가 또 할당 포인트
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

# ============================================================
#  Batched multi-path hopping — 57 kernel launches → max_steps
# ============================================================

function _cudakernel_batched_hop_step!(
    dst, src, U1, U2, U3, U4,
    dirs,         # CuDeviceArray{Int8,2} (max_steps, npaths)
    nsteps_arr,   # CuDeviceArray{Int32,1} (npaths,)sync_W_to_gpu!
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

25개 fused_hopping_gpu! (57 커널) → max_steps 커널.
Double-buffering: hopped_all ↔ temp_5d, step별 1 커널.
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

DeltaRule! 역경로 hop을 batched로 수행.
ci_all을 in-place double-buffering으로 처리 (ci_all ↔ temp_5d).
결과는 ci_all에 저장됨.
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
