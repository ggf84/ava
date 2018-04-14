use real::Real;
use sys::particles::Particle;
use super::{Aligned, Kernel, loop1, loop2, loop3, TILE};

struct Jrk {}
impl Kernel for Jrk {
    type SrcType = [Particle];
    type DstType = [([Real; 3], [Real; 3])];

    const NTILES: usize = 32;

    // flop count: 55
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

        let mut _ieps: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _imass: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _ir0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ir1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ip = &isrc[TILE * ii + i];
                    _ieps[ii][i] = ip.eps;
                    _imass[ii][i] = ip.mass;
                    loop1(3, |k| _ir0[ii][k][i] = ip.pos[k]);
                    loop1(3, |k| _ir1[ii][k][i] = ip.vel[k]);
                }
            }
        }
        let mut _jeps: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jmass: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jr0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _jr1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let jp = &jsrc[TILE * jj + j];
                    _jeps[jj][j] = jp.eps;
                    _jmass[jj][j] = jp.mass;
                    loop1(3, |k| _jr0[jj][k][j] = jp.pos[k]);
                    loop1(3, |k| _jr1[jj][k][j] = jp.vel[k]);
                }
            }
        }

        let mut dr0: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut dr1: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut s0: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s1: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv2: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv3: Aligned<[[Real; TILE]; TILE]> = Default::default();

        for ii in 0..ni_tiles {
            let ieps = &_ieps[ii];
            let imass = &_imass[ii];
            let ir0 = &_ir0[ii];
            let ir1 = &_ir1[ii];
            let ia0 = &mut _ia0[ii];
            let ia1 = &mut _ia1[ii];
            for jj in 0..nj_tiles {
                let jeps = &_jeps[jj];
                let jmass = &_jmass[jj];
                let jr0 = &_jr0[jj];
                let jr1 = &_jr1[jj];
                let ja0 = &mut _ja0[jj];
                let ja1 = &mut _ja1[jj];
                loop3(3, TILE, TILE, |k, i, j| {
                    dr0[k][i][j] = ir0[k][j ^ i] - jr0[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr1[k][i][j] = ir1[k][j ^ i] - jr1[k][j];
                });

                loop2(TILE, TILE, |i, j| {
                    s0[i][j] = ieps[j ^ i] * jeps[j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s0[i][j] += dr0[k][i][j] * dr0[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] = imass[j ^ i] * jmass[j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] = s0[i][j].recip();
                });

                loop2(TILE, TILE, |i, j| {
                    s1[i][j] = 0.0;
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s1[i][j] += dr0[k][i][j] * dr1[k][i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] *= rinv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] *= rinv2[i][j].sqrt();
                });

                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] *= 3.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s1[i][j] *= rinv2[i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    dr1[k][i][j] -= s1[i][j] * dr0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia0[k][j ^ i] -= rinv3[i][j] * dr0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja0[k][j] += rinv3[i][j] * dr0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia1[k][j ^ i] -= rinv3[i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja1[k][j] += rinv3[i][j] * dr1[k][i][j];
                });
            } // jj
        } // ii

        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let minv = 1.0 / _jmass[jj][j];
                    let jacc = &mut jdst[TILE * jj + j];
                    loop1(3, |k| jacc.0[k] += _ja0[jj][k][j] * minv);
                    loop1(3, |k| jacc.1[k] += _ja1[jj][k][j] * minv);
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let minv = 1.0 / _imass[ii][i];
                    let iacc = &mut idst[TILE * ii + i];
                    loop1(3, |k| iacc.0[k] += _ia0[ii][k][i] * minv);
                    loop1(3, |k| iacc.1[k] += _ia1[ii][k][i] * minv);
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
