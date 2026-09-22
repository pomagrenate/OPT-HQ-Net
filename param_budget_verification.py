def dwsep(in_c, out_c, k=3):
    dw = in_c*k*k
    bn1 = 2*in_c
    pw = in_c*out_c
    bn2 = 2*out_c
    return dw+bn1+pw+bn2

def sca(c, r=8):
    hid = max(4, c//r)
    return 2*(c*hid)

def film(edge_c, out_c):
    return edge_c*out_c*2

def pointwise(in_c, out_c):
    return in_c*out_c + 2*out_c

def mhla(dim, heads=4):
    # multi-head linear attention, still dim->dim total proj (heads just split channels)
    return 4*dim*dim

def build(widths, bw, n_blocks_per_stage, n_dilated, stem_out):
    total = 0
    total += dwsep(2, stem_out)
    prev = stem_out
    enc=[]
    for w in widths:
        s = 0
        s += dwsep(prev, w)
        for _ in range(n_blocks_per_stage-1):
            s += dwsep(w, w)
        s += sca(w)
        enc.append(s); total+=s
        prev = w
    proj = pointwise(prev, bw); total+=proj
    bneck = 0
    for _ in range(n_dilated):
        bneck += dwsep(bw, bw)
    bneck += film(1, bw)
    bneck += mhla(bw)
    bneck += mhla(bw)  # 2 attention blocks
    total += bneck
    dec_in = bw
    dec=[]
    for w in reversed(widths):
        concat_c = dec_in + w
        fuse = pointwise(concat_c, w)
        refine = 0
        for _ in range(n_blocks_per_stage-1):
            refine += dwsep(w,w)
        s = sca(w)
        stage_total = fuse+refine+s
        dec.append(stage_total); total += stage_total
        dec_in = w
    head = widths[0]*8 + 8 + 8*1+1  # small 2-layer head: w->8 ->1
    total += head
    return total, enc, dec, proj, bneck, head

for widths, bw, nb, nd, stem in [
    ([24,32,48,64], 96, 2, 4, 20),
    ([32,48,64,96], 128, 2, 6, 24),
    ([32,48,72,96], 144, 2, 6, 24),
    ([40,64,96,128], 160, 2, 6, 32),
]:
    total, enc, dec, proj, bneck, head = build(widths, bw, nb, nd, stem)
    print(widths, bw, nb, nd, '->', f'{total:,}')
