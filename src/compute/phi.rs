use real::Real;
use sys::particles::Particle;
use super::{Aligned, Kernel, loop1, loop2, loop3, TILE};

struct Phi {}
impl Kernel for Phi {
    type SrcType = [Particle];
    type DstType = [Real];

    const NTILES: usize = 32;

    // flop count: 16
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
        let mut _iphi: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ip = &isrc[TILE * ii + i];
                    _ieps[ii][i] = ip.eps;
                    _imass[ii][i] = ip.mass;
                    loop1(3, |k| _ir0[ii][k][i] = ip.pos[k]);
                }
            }
        }
        let mut _jeps: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jmass: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jr0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _jphi: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let jp = &jsrc[TILE * jj + j];
                    _jeps[jj][j] = jp.eps;
                    _jmass[jj][j] = jp.mass;
                    loop1(3, |k| _jr0[jj][k][j] = jp.pos[k]);
                }
            }
        }

        let mut dr0: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut s0: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv1: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv2: Aligned<[[Real; TILE]; TILE]> = Default::default();

        for ii in 0..ni_tiles {
            let ieps = &_ieps[ii];
            let imass = &_imass[ii];
            let ir0 = &_ir0[ii];
            let iphi = &mut _iphi[ii];
            for jj in 0..nj_tiles {
                let jeps = &_jeps[jj];
                let jmass = &_jmass[jj];
                let jr0 = &_jr0[jj];
                let jphi = &mut _jphi[jj];
                loop3(3, TILE, TILE, |k, i, j| {
                    dr0[k][i][j] = ir0[k][j ^ i] - jr0[k][j];
                });

                loop2(TILE, TILE, |i, j| {
                    s0[i][j] = ieps[j ^ i] * jeps[j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s0[i][j] += dr0[k][i][j] * dr0[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv1[i][j] = imass[j ^ i] * jmass[j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] = s0[i][j].recip();
                });

                loop2(TILE, TILE, |i, j| {
                    rinv1[i][j] *= rinv2[i][j].sqrt();
                });

                loop2(TILE, TILE, |i, j| {
                    iphi[j ^ i] -= rinv1[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    jphi[j] -= rinv1[i][j];
                });
            } // jj
        } // ii

        for jj in 0..nj_tiles {
            for j in 0..TILE {
                if (TILE * jj + j) < nj {
                    let minv = 1.0 / _jmass[jj][j];
                    let jphi = &mut jdst[TILE * jj + j];
                    *jphi += _jphi[jj][j] * minv;
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let minv = 1.0 / _imass[ii][i];
                    let iphi = &mut idst[TILE * ii + i];
                    *iphi += _iphi[ii][i] * minv;
                }
            }
        }
    }
}

pub fn triangle(src: &[Particle]) -> Vec<Real> {
    let mut dst = vec![0.0; src.len()];
    Phi {}.triangle(&src[..], &mut dst[..]);
    dst
}

pub fn rectangle(isrc: &[Particle], jsrc: &[Particle]) -> (Vec<Real>, Vec<Real>) {
    let mut idst = vec![0.0; isrc.len()];
    let mut jdst = vec![0.0; jsrc.len()];
    Phi {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);
    (idst, jdst)
}

// -- end of file --
