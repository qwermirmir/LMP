# Index arithmetic for the blocked GPU lattice layout of Gaugefields and
# LatticeDiracOperators: neighbour sites across block boundaries and the
# boundary-condition sign.

# ============================================================
#  Fused GPU hop kernels — 3 kernels per step → 1
# ============================================================

# Inline helper functions from Gaugefields blockinfo

@inline _cvt_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)

@inline function _si_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b; rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == blocks - 1
        bs = bs - (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == blocknumbers - 1
            rs = rs - (blocknumbers - 1) * blocknumbers_s
        else
            rs += blocknumbers_s
        end
    else
        bs += blocks_s
    end
    return bs, rs
end

@inline function _si_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b; rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == 0
        bs = bs + (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == 0
            rs = rs + (blocknumbers - 1) * blocknumbers_s
        else
            rs -= blocknumbers_s
        end
    else
        bs -= blocks_s
    end
    return bs, rs
end

@inline function _shifted_idx(b, r, shift, blockinfo)
    bs = b; rs = r
    for idim = 1:4
        blk   = blockinfo.blocks[idim]
        blk_s = blockinfo.blocks_s[idim]
        bn    = blockinfo.blocknumbers[idim]
        bn_s  = blockinfo.blocknumbers_s[idim]
        ns = abs(shift[idim])
        if shift[idim] > 0
            for _ = 1:ns
                bs, rs = _si_plus(bs, rs, blk, blk_s, bn, bn_s)
            end
        else
            for _ = 1:ns
                bs, rs = _si_minus(bs, rs, blk, blk_s, bn, bn_s)
            end
        end
    end
    return bs, rs
end

@inline function _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix = 1 + _cvt_x(b, blockinfo.blocks[1], blockinfo.blocks_s[1]) +
             _cvt_x(r, blockinfo.blocknumbers[1], blockinfo.blocknumbers_s[1]) * blockinfo.blocks[1]
    iy = 1 + _cvt_x(b, blockinfo.blocks[2], blockinfo.blocks_s[2]) +
             _cvt_x(r, blockinfo.blocknumbers[2], blockinfo.blocknumbers_s[2]) * blockinfo.blocks[2]
    iz = 1 + _cvt_x(b, blockinfo.blocks[3], blockinfo.blocks_s[3]) +
             _cvt_x(r, blockinfo.blocknumbers[3], blockinfo.blocknumbers_s[3]) * blockinfo.blocks[3]
    it = 1 + _cvt_x(b, blockinfo.blocks[4], blockinfo.blocks_s[4]) +
             _cvt_x(r, blockinfo.blocknumbers[4], blockinfo.blocknumbers_s[4]) * blockinfo.blocks[4]
    f = ifelse(ix + shift[1] > NX || ix + shift[1] < 1, bc[1], Int8(1)) *
        ifelse(iy + shift[2] > NY || iy + shift[2] < 1, bc[2], Int8(1)) *
        ifelse(iz + shift[3] > NZ || iz + shift[3] < 1, bc[3], Int8(1)) *
        ifelse(it + shift[4] > NT || it + shift[4] < 1, bc[4], Int8(1))
    return f
end

const _BC_DEFAULT = (Int8(1), Int8(1), Int8(1), Int8(-1))
