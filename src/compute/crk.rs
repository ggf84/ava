use super::{Kernel, loop1, loop2, loop3, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
pub struct CrkData {
    pub eps: [Real; TILE],
    pub mass: [Real; TILE],
    pub r0: [[Real; TILE]; 3],
    pub r1: [[Real; TILE]; 3],
    pub r2: [[Real; TILE]; 3],
    pub r3: [[Real; TILE]; 3],
    pub a0: [[Real; TILE]; 3],
    pub a1: [[Real; TILE]; 3],
    pub a2: [[Real; TILE]; 3],
    pub a3: [[Real; TILE]; 3],
}

pub struct Crk {}
impl Crk {
    // flop count: 157
    pub fn p2p(&self, ip: &mut CrkData, jp: &mut CrkData) {
        const CQ21: Real = 5.0 / 3.0;
        const CQ31: Real = 8.0 / 3.0;
        const CQ32: Real = 7.0 / 3.0;
        let mut dr0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut dr1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut dr2: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut dr3: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s01: [[Real; TILE]; TILE] = Default::default();
        let mut s02: [[Real; TILE]; TILE] = Default::default();
        let mut s03: [[Real; TILE]; TILE] = Default::default();
        let mut s11: [[Real; TILE]; TILE] = Default::default();
        let mut s12: [[Real; TILE]; TILE] = Default::default();
        let mut q01: [[Real; TILE]; TILE] = Default::default();
        let mut q02: [[Real; TILE]; TILE] = Default::default();
        let mut q03: [[Real; TILE]; TILE] = Default::default();
        let mut q12: [[Real; TILE]; TILE] = Default::default();
        let mut q13: [[Real; TILE]; TILE] = Default::default();
        let mut q23: [[Real; TILE]; TILE] = Default::default();
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
        loop3(3, TILE, TILE, |k, i, j| {
            dr2[k][i][j] = ip.r2[k][j ^ i] - jp.r2[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr3[k][i][j] = ip.r3[k][j ^ i] - jp.r3[k][j];
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
            s02[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s02[i][j] += dr0[k][i][j] * dr2[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s11[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s11[i][j] += dr1[k][i][j] * dr1[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s02[i][j] += s11[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            s03[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s03[i][j] += dr0[k][i][j] * dr3[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s12[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s12[i][j] += dr1[k][i][j] * dr2[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s03[i][j] += 3.0 * s12[i][j];
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
        loop2(TILE, TILE, |i, j| {
            q02[i][j] = rinv2[i][j] * s02[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            q03[i][j] = rinv2[i][j] * s03[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            q03[i][j] -= (CQ31 * q02[i][j]) * q01[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            q02[i][j] -= (CQ21 * q01[i][j]) * q01[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            q03[i][j] -= (CQ32 * q01[i][j]) * q02[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            q12[i][j] = 2.0 * q01[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            q13[i][j] = 3.0 * q02[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            q23[i][j] = 3.0 * q01[i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            dr3[k][i][j] -= q23[i][j] * dr2[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr3[k][i][j] -= q13[i][j] * dr1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr3[k][i][j] -= q03[i][j] * dr0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr2[k][i][j] -= q12[i][j] * dr1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            dr2[k][i][j] -= q02[i][j] * dr0[k][i][j];
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

        loop3(3, TILE, TILE, |k, i, j| {
            ip.a2[k][j ^ i] -= mm_r3[i][j] * dr2[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp.a2[k][j] += mm_r3[i][j] * dr2[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip.a3[k][j ^ i] -= mm_r3[i][j] * dr3[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp.a3[k][j] += mm_r3[i][j] * dr3[k][i][j];
        });
    }
}

impl Kernel for Crk {
    type SrcType = [Particle];
    type DstType = [([Real; 3], [Real; 3], [Real; 3], [Real; 3])];

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

        let mut idata: [CrkData; Self::NTILES] = [Default::default(); Self::NTILES];
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ip = &isrc[TILE * ii + i];
                    idata[ii].eps[i] = ip.eps;
                    idata[ii].mass[i] = ip.mass;
                    loop1(3, |k| idata[ii].r0[k][i] = ip.pos[k]);
                    loop1(3, |k| idata[ii].r1[k][i] = ip.vel[k]);
                    loop1(3, |k| idata[ii].r2[k][i] = ip.acc0[k]);
                    loop1(3, |k| idata[ii].r3[k][i] = ip.acc1[k]);
                }
            }
        }
        let mut jdata: [CrkData; Self::NTILES] = [Default::default(); Self::NTILES];
        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let jp = &jsrc[TILE * jj + j];
                    jdata[jj].eps[j] = jp.eps;
                    jdata[jj].mass[j] = jp.mass;
                    loop1(3, |k| jdata[jj].r0[k][j] = jp.pos[k]);
                    loop1(3, |k| jdata[jj].r1[k][j] = jp.vel[k]);
                    loop1(3, |k| jdata[jj].r2[k][j] = jp.acc0[k]);
                    loop1(3, |k| jdata[jj].r3[k][j] = jp.acc1[k]);
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
                    loop1(3, |k| jacc.2[k] += jdata[jj].a2[k][j] * minv);
                    loop1(3, |k| jacc.3[k] += jdata[jj].a3[k][j] * minv);
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
                    loop1(3, |k| iacc.2[k] += idata[ii].a2[k][i] * minv);
                    loop1(3, |k| iacc.3[k] += idata[ii].a3[k][i] * minv);
                }
            }
        }
    }
}

pub fn triangle(
    src: &[Particle],
) -> (
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
) {
    let mut dst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); src.len()];
    Crk {}.triangle(&src[..], &mut dst[..]);

    let mut acc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    dst.iter().for_each(|a| {
        acc.0.push(a.0);
        acc.1.push(a.1);
        acc.2.push(a.2);
        acc.3.push(a.3);
    });
    acc
}

pub fn rectangle(
    isrc: &[Particle],
    jsrc: &[Particle],
) -> (
    (
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
    ),
    (
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
    ),
) {
    let mut idst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); isrc.len()];
    let mut jdst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); jsrc.len()];
    Crk {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);

    let mut iacc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut jacc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    idst.iter().for_each(|ia| {
        iacc.0.push(ia.0);
        iacc.1.push(ia.1);
        iacc.2.push(ia.2);
        iacc.3.push(ia.3);
    });
    jdst.iter().for_each(|ja| {
        jacc.0.push(ja.0);
        jacc.1.push(ja.1);
        jacc.2.push(ja.2);
        jacc.3.push(ja.3);
    });
    (iacc, jacc)
}

// -- end of file --
