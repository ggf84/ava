use super::{loop1, loop2, loop3, FromSoA, ToSoA, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct CrkSrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
    rdot1: [[Real; TILE]; 3],
    rdot2: [[Real; TILE]; 3],
    rdot3: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct CrkDstSoA {
    adot0: [[Real; TILE]; 3],
    adot1: [[Real; TILE]; 3],
    adot2: [[Real; TILE]; 3],
    adot3: [[Real; TILE]; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct CrkSrc {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
    rdot1: [Real; 3],
    rdot2: [Real; 3],
    rdot3: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct CrkDst {
    adot0: [Real; 3],
    adot1: [Real; 3],
    adot2: [Real; 3],
    adot3: [Real; 3],
}

impl<'a> ToSoA for CrkSrcSlice<'a> {
    type SrcTypeSoA = CrkSrcSoA;
    fn to_soa(&self, p_src: &mut [Self::SrcTypeSoA]) {
        let n = self.len();
        let n_tiles = p_src.len();
        for jj in 0..n_tiles {
            for j in 0..TILE {
                let jjj = TILE * jj + j;
                if jjj < n {
                    p_src[jj].eps[j] = self.eps[jjj];
                    p_src[jj].mass[j] = self.mass[jjj];
                    loop1(3, |k| p_src[jj].rdot0[k][j] = self.rdot0[jjj][k]);
                    loop1(3, |k| p_src[jj].rdot1[k][j] = self.rdot1[jjj][k]);
                    loop1(3, |k| p_src[jj].rdot2[k][j] = self.rdot2[jjj][k]);
                    loop1(3, |k| p_src[jj].rdot3[k][j] = self.rdot3[jjj][k]);
                }
            }
        }
    }
}

impl<'a> FromSoA for CrkDstSliceMut<'a> {
    type SrcTypeSoA = CrkSrcSoA;
    type DstTypeSoA = CrkDstSoA;
    fn from_soa(&mut self, p_src: &[Self::SrcTypeSoA], p_dst: &[Self::DstTypeSoA]) {
        let n = self.len();
        let n_tiles = p_src.len();
        for jj in 0..n_tiles {
            for j in 0..TILE {
                let jjj = TILE * jj + j;
                if jjj < n {
                    let minv = 1.0 / p_src[jj].mass[j];
                    loop1(3, |k| self.adot0[jjj][k] += p_dst[jj].adot0[k][j] * minv);
                    loop1(3, |k| self.adot1[jjj][k] += p_dst[jj].adot1[k][j] * minv);
                    loop1(3, |k| self.adot2[jjj][k] += p_dst[jj].adot2[k][j] * minv);
                    loop1(3, |k| self.adot3[jjj][k] += p_dst[jj].adot3[k][j] * minv);
                }
            }
        }
    }
}

pub struct Crk {}
impl_kernel!(CrkSrcSlice, CrkDstSliceMut, CrkSrcSoA, CrkDstSoA, 64);

impl Kernel for Crk {
    // flop count: 157
    fn p2p(
        &self,
        ip_src: &CrkSrcSoA,
        ip_dst: &mut CrkDstSoA,
        jp_src: &CrkSrcSoA,
        jp_dst: &mut CrkDstSoA,
    ) {
        const CQ21: Real = 5.0 / 3.0;
        const CQ31: Real = 8.0 / 3.0;
        const CQ32: Real = 7.0 / 3.0;
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot2: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot3: [[[Real; TILE]; TILE]; 3] = Default::default();
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
            drdot0[k][i][j] = ip_src.rdot0[k][j ^ i] - jp_src.rdot0[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot1[k][i][j] = ip_src.rdot1[k][j ^ i] - jp_src.rdot1[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot2[k][i][j] = ip_src.rdot2[k][j ^ i] - jp_src.rdot2[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot3[k][i][j] = ip_src.rdot3[k][j ^ i] - jp_src.rdot3[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += drdot0[k][i][j] * drdot0[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            s01[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s01[i][j] += drdot0[k][i][j] * drdot1[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            s02[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s02[i][j] += drdot0[k][i][j] * drdot2[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s11[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s11[i][j] += drdot1[k][i][j] * drdot1[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s02[i][j] += s11[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            s03[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s03[i][j] += drdot0[k][i][j] * drdot3[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s12[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s12[i][j] += drdot1[k][i][j] * drdot2[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s03[i][j] += 3.0 * s12[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip_src.mass[j ^ i] * jp_src.mass[j];
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
            drdot3[k][i][j] -= q23[i][j] * drdot2[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot3[k][i][j] -= q13[i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot3[k][i][j] -= q03[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot2[k][i][j] -= q12[i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot2[k][i][j] -= q02[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot1[k][i][j] -= q01[i][j] * drdot0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot0[k][j ^ i] -= mm_r3[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot0[k][j] += mm_r3[i][j] * drdot0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot1[k][j ^ i] -= mm_r3[i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot1[k][j] += mm_r3[i][j] * drdot1[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot2[k][j ^ i] -= mm_r3[i][j] * drdot2[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot2[k][j] += mm_r3[i][j] * drdot2[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot3[k][j ^ i] -= mm_r3[i][j] * drdot3[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot3[k][j] += mm_r3[i][j] * drdot3[k][i][j];
        });
    }
}

pub fn triangle(
    psys: &[Particle],
) -> (
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
) {
    let mut src = CrkSrcVec::with_capacity(psys.len());
    let mut dst = CrkDstVec::with_capacity(psys.len());
    for p in psys.iter() {
        src.push(CrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
            rdot2: p.acc0,
            rdot3: p.acc1,
        });
        dst.push(Default::default());
    }

    Crk {}.triangle(&src.as_slice(), &mut dst.as_mut_slice());

    (dst.adot0, dst.adot1, dst.adot2, dst.adot3)
}

pub fn rectangle(
    ipsys: &[Particle],
    jpsys: &[Particle],
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
    let mut isrc = CrkSrcVec::with_capacity(ipsys.len());
    let mut idst = CrkDstVec::with_capacity(ipsys.len());
    for p in ipsys.iter() {
        isrc.push(CrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
            rdot2: p.acc0,
            rdot3: p.acc1,
        });
        idst.push(Default::default());
    }

    let mut jsrc = CrkSrcVec::with_capacity(jpsys.len());
    let mut jdst = CrkDstVec::with_capacity(jpsys.len());
    for p in jpsys.iter() {
        jsrc.push(CrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
            rdot2: p.acc0,
            rdot3: p.acc1,
        });
        jdst.push(Default::default());
    }

    Crk {}.rectangle(
        &isrc.as_slice(),
        &mut idst.as_mut_slice(),
        &jsrc.as_slice(),
        &mut jdst.as_mut_slice(),
    );

    (
        (idst.adot0, idst.adot1, idst.adot2, idst.adot3),
        (jdst.adot0, jdst.adot1, jdst.adot2, jdst.adot3),
    )
}

#[cfg(all(feature = "nightly-bench", test))]
mod bench {
    extern crate test;

    use self::test::Bencher;
    use super::*;
    use rand::{
        distributions::{Distribution, Standard},
        Rng, SeedableRng, StdRng,
    };

    const NTILES: usize = 256 / TILE;

    impl Distribution<CrkSrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> CrkSrcSoA {
            CrkSrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
                rdot1: rng.gen(),
                rdot2: rng.gen(),
                rdot3: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Crk {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [CrkSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [CrkDstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [CrkSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [CrkDstSoA; NTILES] = [Default::default(); NTILES];
            ip_src.iter_mut().for_each(|p| *p = rng.gen());
            jp_src.iter_mut().for_each(|p| *p = rng.gen());
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&ip_src[ii], &mut ip_dst[ii], &jp_src[jj], &mut jp_dst[jj]);
                }
            }
            (ip_dst, jp_dst)
        });
    }
}

// -- end of file --
