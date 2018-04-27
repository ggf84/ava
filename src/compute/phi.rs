use real::Real;
use sys::particles::Particle;
use super::{Kernel, loop1, loop2, loop3, TILE};

#[repr(align(16))]
#[derive(Debug, Default)]
struct PhiData {
    eps: [Real; TILE],
    mass: [Real; TILE],
    r0: [[Real; TILE]; 3],
    phi: [Real; TILE],
}

struct Phi {}
impl Phi {
    // flop count: 16
    fn p2p(&self, ip: &mut PhiData, jp: &mut PhiData) {
        let mut dr0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s0: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();

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
            rinv1[i][j] = ip.mass[j ^ i] * jp.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] = s0[i][j].recip();
        });

        loop2(TILE, TILE, |i, j| {
            rinv1[i][j] *= rinv2[i][j].sqrt();
        });

        loop2(TILE, TILE, |i, j| {
            ip.phi[j ^ i] -= rinv1[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            jp.phi[j] -= rinv1[i][j];
        });
    }
}

impl Kernel for Phi {
    type SrcType = [Particle];
    type DstType = [Real];

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

        let mut idata: [PhiData; Self::NTILES] = Default::default();
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
        let mut jdata: [PhiData; Self::NTILES] = Default::default();
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
                    let jphi = &mut jdst[TILE * jj + j];
                    *jphi += jdata[jj].phi[j] * minv;
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let minv = 1.0 / idata[ii].mass[i];
                    let iphi = &mut idst[TILE * ii + i];
                    *iphi += idata[ii].phi[i] * minv;
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
