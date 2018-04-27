use real::Real;
use sys::particles::Particle;
use super::{Kernel, loop1, loop2, loop3, TILE};

#[repr(align(16))]
#[derive(Debug, Default)]
struct AccData {
    eps: [Real; TILE],
    mass: [Real; TILE],
    r0: [[Real; TILE]; 3],
    a0: [[Real; TILE]; 3],
}

struct Acc {}
impl Acc {
    // flop count: 27
    fn p2p(&self, ip: &mut AccData, jp: &mut AccData) {
        let mut dr0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s0: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut rinv3: [[Real; TILE]; TILE] = Default::default();

        loop3(3, TILE, TILE, |k, i, j| {
            dr0[k][i][j] = ip.r0[k][j ^ i] - jp.r0[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s0[i][j] = ip.eps[j ^ i] * jp.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s0[i][j] += dr0[k][i][j] * dr0[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv3[i][j] = ip.mass[j ^ i] * jp.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] = s0[i][j].recip();
        });

        loop2(TILE, TILE, |i, j| {
            rinv3[i][j] *= rinv2[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv3[i][j] *= rinv2[i][j].sqrt();
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip.a0[k][j ^ i] -= rinv3[i][j] * dr0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp.a0[k][j] += rinv3[i][j] * dr0[k][i][j];
        });
    }
}

impl Kernel for Acc {
    type SrcType = [Particle];
    type DstType = [([Real; 3],)];

    const NTILES: usize = 32;

    fn kernel(
        &self,
        isrc: &Self::SrcType,
        jsrc: &Self::SrcType,
        idst: &mut Self::DstType,
        jdst: &mut Self::DstType,
    ) {
        let ni = isrc.len();
        let nj = jsrc.len();
        let ni_tiles = (ni + TILE - 1) / TILE;
        let nj_tiles = (nj + TILE - 1) / TILE;

        let mut idata: [AccData; Self::NTILES] = Default::default();
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ip = &isrc[TILE * ii + i];
                    idata[ii].eps[i] = ip.eps;
                    idata[ii].mass[i] = ip.mass;
                    loop1(3, |k| idata[ii].r0[k][i] = ip.pos[k]);
                }
            }
        }
        let mut jdata: [AccData; Self::NTILES] = Default::default();
        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let jp = &jsrc[TILE * jj + j];
                    jdata[jj].eps[j] = jp.eps;
                    jdata[jj].mass[j] = jp.mass;
                    loop1(3, |k| jdata[jj].r0[k][j] = jp.pos[k]);
                }
            }
        }

        for ii in 0..ni_tiles {
            for jj in 0..nj_tiles {
                self.p2p(&mut idata[ii], &mut jdata[jj]);
            }
        }

        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let minv = 1.0 / jdata[jj].mass[j];
                    let jacc = &mut jdst[TILE * jj + j];
                    loop1(3, |k| jacc.0[k] += jdata[jj].a0[k][j] * minv);
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let minv = 1.0 / idata[ii].mass[i];
                    let iacc = &mut idst[TILE * ii + i];
                    loop1(3, |k| iacc.0[k] += idata[ii].a0[k][i] * minv);
                }
            }
        }
    }
}

pub fn triangle(src: &[Particle]) -> (Vec<[Real; 3]>,) {
    let mut dst = vec![([0.0; 3],); src.len()];
    Acc {}.triangle(&src[..], &mut dst[..]);

    let mut acc = (Vec::new(),);
    dst.iter().for_each(|a| {
        acc.0.push(a.0);
    });
    acc
}

pub fn rectangle(isrc: &[Particle], jsrc: &[Particle]) -> ((Vec<[Real; 3]>,), (Vec<[Real; 3]>,)) {
    let mut idst = vec![([0.0; 3],); isrc.len()];
    let mut jdst = vec![([0.0; 3],); jsrc.len()];
    Acc {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);

    let mut iacc = (Vec::new(),);
    let mut jacc = (Vec::new(),);
    idst.iter().for_each(|ia| {
        iacc.0.push(ia.0);
    });
    jdst.iter().for_each(|ja| {
        jacc.0.push(ja.0);
    });
    (iacc, jacc)
}

// -- end of file --
