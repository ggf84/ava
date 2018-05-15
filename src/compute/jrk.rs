use super::{Kernel, loop1, loop2, loop3, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
pub struct JrkData {
    pub eps: [Real; TILE],
    pub mass: [Real; TILE],
    pub r0: [[Real; TILE]; 3],
    pub r1: [[Real; TILE]; 3],
    pub a0: [[Real; TILE]; 3],
    pub a1: [[Real; TILE]; 3],
}

pub struct Jrk {}
impl Jrk {
    // flop count: 56
    pub fn p2p(&self, ip: &mut JrkData, jp: &mut JrkData) {
        let mut dr0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut dr1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s01: [[Real; TILE]; TILE] = Default::default();
        let mut q01: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

        loop3(3, TILE, TILE, |k, i, j| {
            dr0[k][i][j] = ip.r0[k][j ^ i] - jp.r0[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr1[k][i][j] = ip.r1[k][j ^ i] - jp.r1[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip.eps[j ^ i] * jp.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += dr0[k][i][j] * dr0[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            s01[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s01[i][j] += dr0[k][i][j] * dr1[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip.mass[j ^ i] * jp.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] = s00[i][j].recip();
        });
        loop2(TILE, TILE, |i, j| {
            rinv1[i][j] = rinv2[i][j].sqrt();
        });
        loop2(TILE, TILE, |i, j| {
            mm_r3[i][j] = mm[i][j] * rinv2[i][j] * rinv1[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] *= 3.0;
        });

        loop2(TILE, TILE, |i, j| {
            q01[i][j] = rinv2[i][j] * s01[i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            dr1[k][i][j] -= q01[i][j] * dr0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip.a0[k][j ^ i] -= mm_r3[i][j] * dr0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp.a0[k][j] += mm_r3[i][j] * dr0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip.a1[k][j ^ i] -= mm_r3[i][j] * dr1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp.a1[k][j] += mm_r3[i][j] * dr1[k][i][j];
        });
    }
}

impl Kernel for Jrk {
    type SrcType = [Particle];
    type DstType = [([Real; 3], [Real; 3])];

    const NTILES: usize = 64 / TILE;

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

        let mut idata: [JrkData; Self::NTILES] = [Default::default(); Self::NTILES];
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ip = &isrc[TILE * ii + i];
                    idata[ii].eps[i] = ip.eps;
                    idata[ii].mass[i] = ip.mass;
                    loop1(3, |k| idata[ii].r0[k][i] = ip.pos[k]);
                    loop1(3, |k| idata[ii].r1[k][i] = ip.vel[k]);
                }
            }
        }
        let mut jdata: [JrkData; Self::NTILES] = [Default::default(); Self::NTILES];
        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let jp = &jsrc[TILE * jj + j];
                    jdata[jj].eps[j] = jp.eps;
                    jdata[jj].mass[j] = jp.mass;
                    loop1(3, |k| jdata[jj].r0[k][j] = jp.pos[k]);
                    loop1(3, |k| jdata[jj].r1[k][j] = jp.vel[k]);
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
                    loop1(3, |k| jacc.1[k] += jdata[jj].a1[k][j] * minv);
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let minv = 1.0 / idata[ii].mass[i];
                    let iacc = &mut idst[TILE * ii + i];
                    loop1(3, |k| iacc.0[k] += idata[ii].a0[k][i] * minv);
                    loop1(3, |k| iacc.1[k] += idata[ii].a1[k][i] * minv);
                }
            }
        }
    }
}

pub fn triangle(src: &[Particle]) -> (Vec<[Real; 3]>, Vec<[Real; 3]>) {
    let mut dst = vec![([0.0; 3], [0.0; 3]); src.len()];
    Jrk {}.triangle(&src[..], &mut dst[..]);

    let mut acc = (Vec::new(), Vec::new());
    dst.iter().for_each(|a| {
        acc.0.push(a.0);
        acc.1.push(a.1);
    });
    acc
}

pub fn rectangle(
    isrc: &[Particle],
    jsrc: &[Particle],
) -> (
    (Vec<[Real; 3]>, Vec<[Real; 3]>),
    (Vec<[Real; 3]>, Vec<[Real; 3]>),
) {
    let mut idst = vec![([0.0; 3], [0.0; 3]); isrc.len()];
    let mut jdst = vec![([0.0; 3], [0.0; 3]); jsrc.len()];
    Jrk {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);

    let mut iacc = (Vec::new(), Vec::new());
    let mut jacc = (Vec::new(), Vec::new());
    idst.iter().for_each(|ia| {
        iacc.0.push(ia.0);
        iacc.1.push(ia.1);
    });
    jdst.iter().for_each(|ja| {
        jacc.0.push(ja.0);
        jacc.1.push(ja.1);
    });
    (iacc, jacc)
}

// -- end of file --
