use super::{Kernel, loop1, loop2, loop3, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
pub struct EnergyData {
    pub eps: [Real; TILE],
    pub mass: [Real; TILE],
    pub r0: [[Real; TILE]; 3],
    pub r1: [[Real; TILE]; 3],
    pub ekin: [Real; TILE],
    pub epot: [Real; TILE],
}

pub struct Energy {}
impl Energy {
    // flop count: 28
    pub fn p2p(&self, ip: &mut EnergyData, jp: &mut EnergyData) {
        let mut dr0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut dr1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s11: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mmv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r1: [[Real; TILE]; TILE] = Default::default();

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
            rinv2[i][j] = s00[i][j].recip();
        });
        loop2(TILE, TILE, |i, j| {
            rinv1[i][j] = rinv2[i][j].sqrt();
        });

        loop2(TILE, TILE, |i, j| {
            s11[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s11[i][j] += dr1[k][i][j] * dr1[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip.mass[j ^ i] * jp.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            mmv2[i][j] = mm[i][j] * s11[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            mm_r1[i][j] = mm[i][j] * rinv1[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            ip.ekin[j ^ i] += mmv2[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            jp.ekin[j] += mmv2[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            ip.epot[j ^ i] -= mm_r1[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            jp.epot[j] -= mm_r1[i][j];
        });
    }
}

impl Kernel for Energy {
    type SrcType = [Particle];
    type DstType = [(Real, Real)];

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

        let mut idata: [EnergyData; Self::NTILES] = [Default::default(); Self::NTILES];
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
        let mut jdata: [EnergyData; Self::NTILES] = [Default::default(); Self::NTILES];
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
                    let jenergy = &mut jdst[TILE * jj + j];
                    jenergy.0 += jdata[jj].ekin[j];
                    jenergy.1 += jdata[jj].epot[j];
                }
            }
        }
        for ii in 0..ni_tiles {
            for i in 0..TILE {
                if (TILE * ii + i) < ni {
                    let ienergy = &mut idst[TILE * ii + i];
                    ienergy.0 += idata[ii].ekin[i];
                    ienergy.1 += idata[ii].epot[i];
                }
            }
        }
    }
}

pub fn triangle(src: &[Particle]) -> (Vec<Real>, Vec<Real>) {
    let mut dst = vec![(0.0, 0.0); src.len()];
    Energy {}.triangle(&src[..], &mut dst[..]);

    let mut energy = (Vec::new(), Vec::new());
    dst.iter().for_each(|e| {
        energy.0.push(e.0);
        energy.1.push(e.1);
    });
    energy
}

pub fn rectangle(
    isrc: &[Particle],
    jsrc: &[Particle],
) -> ((Vec<Real>, Vec<Real>), (Vec<Real>, Vec<Real>)) {
    let mut idst = vec![(0.0, 0.0); isrc.len()];
    let mut jdst = vec![(0.0, 0.0); jsrc.len()];
    Energy {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);

    let mut ienergy = (Vec::new(), Vec::new());
    let mut jenergy = (Vec::new(), Vec::new());
    idst.iter().for_each(|ie| {
        ienergy.0.push(ie.0);
        ienergy.1.push(ie.1);
    });
    jdst.iter().for_each(|je| {
        jenergy.0.push(je.0);
        jenergy.1.push(je.1);
    });
    (ienergy, jenergy)
}

// -- end of file --
