use super::{loop1, loop2, loop3, Compute, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::Particle};
use serde_derive::{Deserialize, Serialize};
use soa_derive::StructOfArray;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct Acc2SrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
    rdot1: [[Real; TILE]; 3],
    rdot2: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct Acc2DstSoA {
    adot0: [[Real; TILE]; 3],
    adot1: [[Real; TILE]; 3],
    adot2: [[Real; TILE]; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct Acc2Src {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
    rdot1: [Real; 3],
    rdot2: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct Acc2Dst {
    adot0: [Real; 3],
    adot1: [Real; 3],
    adot2: [Real; 3],
}

impl<'a> ToSoA<[Acc2SrcSoA]> for Acc2SrcSlice<'a> {
    fn to_soa(&self, ps_src: &mut [Acc2SrcSoA]) {
        let n = self.len();
        let mut jj = 0;
        for p_src in ps_src.iter_mut() {
            for j in 0..TILE {
                if jj < n {
                    p_src.eps[j] = self.eps[jj];
                    p_src.mass[j] = self.mass[jj];
                    loop1(3, |k| p_src.rdot0[k][j] = self.rdot0[jj][k]);
                    loop1(3, |k| p_src.rdot1[k][j] = self.rdot1[jj][k]);
                    loop1(3, |k| p_src.rdot2[k][j] = self.rdot2[jj][k]);
                    jj += 1;
                }
            }
        }
    }
}

impl<'a> FromSoA<[Acc2SrcSoA], [Acc2DstSoA]> for Acc2DstSliceMut<'a> {
    fn from_soa(&mut self, ps_src: &[Acc2SrcSoA], ps_dst: &[Acc2DstSoA]) {
        let n = self.len();
        let mut jj = 0;
        for (p_src, p_dst) in ps_src.iter().zip(ps_dst.iter()) {
            for j in 0..TILE {
                if jj < n {
                    let minv = 1.0 / p_src.mass[j];
                    loop1(3, |k| self.adot0[jj][k] += p_dst.adot0[k][j] * minv);
                    loop1(3, |k| self.adot1[jj][k] += p_dst.adot1[k][j] * minv);
                    loop1(3, |k| self.adot2[jj][k] += p_dst.adot2[k][j] * minv);
                    jj += 1;
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Acc2 {}
impl_kernel!(Acc2SrcSlice, Acc2DstSliceMut, Acc2SrcSoA, Acc2DstSoA, 64);

impl Kernel for Acc2 {
    // flop count: 101
    fn p2p(
        &self,
        ip_src: &Acc2SrcSoA,
        ip_dst: &mut Acc2DstSoA,
        jp_src: &Acc2SrcSoA,
        jp_dst: &mut Acc2DstSoA,
    ) {
        const CQ21: Real = 5.0 / 3.0;
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot2: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s01: [[Real; TILE]; TILE] = Default::default();
        let mut s02: [[Real; TILE]; TILE] = Default::default();
        let mut s11: [[Real; TILE]; TILE] = Default::default();
        let mut q01: [[Real; TILE]; TILE] = Default::default();
        let mut q02: [[Real; TILE]; TILE] = Default::default();
        let mut q12: [[Real; TILE]; TILE] = Default::default();
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

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip_src.mass[j ^ i] * jp_src.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += drdot0[k][i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s01[i][j] += drdot0[k][i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s11[i][j] += drdot1[k][i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s02[i][j] += drdot0[k][i][j] * drdot2[k][i][j];
        });
        loop2(TILE, TILE, |i, j| {
            s02[i][j] += s11[i][j];
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
            q02[i][j] -= (CQ21 * q01[i][j]) * q01[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            q12[i][j] = 2.0 * q01[i][j];
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
    }
}

impl Compute<[Particle]> for Acc2 {
    type Output = (Vec<[Real; 3]>, Vec<[Real; 3]>, Vec<[Real; 3]>);
    /// For each particle of the system, compute the k-th derivatives
    /// of the gravitational acceleration, for k = {0, 1, 2}.
    fn compute(&self, psys: &[Particle]) -> Self::Output {
        let mut src = Acc2SrcVec::with_capacity(psys.len());
        let mut dst = Acc2DstVec::with_capacity(psys.len());
        for p in psys.iter() {
            src.push(Acc2Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
                rdot2: p.acc0,
            });
            dst.push(Default::default());
        }

        self.triangle(&src.as_slice(), &mut dst.as_mut_slice());

        (dst.adot0, dst.adot1, dst.adot2)
    }
    /// For each particle of two disjoint systems, A and B, compute the mutual k-th derivatives
    /// of the gravitational acceleration, for k = {0, 1, 2}.
    fn compute_mutual(
        &self,
        ipsys: &[Particle],
        jpsys: &[Particle],
    ) -> (Self::Output, Self::Output) {
        let mut isrc = Acc2SrcVec::with_capacity(ipsys.len());
        let mut idst = Acc2DstVec::with_capacity(ipsys.len());
        for p in ipsys.iter() {
            isrc.push(Acc2Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
                rdot2: p.acc0,
            });
            idst.push(Default::default());
        }

        let mut jsrc = Acc2SrcVec::with_capacity(jpsys.len());
        let mut jdst = Acc2DstVec::with_capacity(jpsys.len());
        for p in jpsys.iter() {
            jsrc.push(Acc2Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
                rdot2: p.acc0,
            });
            jdst.push(Default::default());
        }

        self.rectangle(
            &isrc.as_slice(),
            &mut idst.as_mut_slice(),
            &jsrc.as_slice(),
            &mut jdst.as_mut_slice(),
        );

        (
            (idst.adot0, idst.adot1, idst.adot2),
            (jdst.adot0, jdst.adot1, jdst.adot2),
        )
    }
}

#[cfg(all(feature = "nightly", test))]
mod bench {
    use super::*;
    use rand::{
        distributions::{Distribution, Standard},
        Rng, SeedableRng, StdRng,
    };
    use test::Bencher;

    const NTILES: usize = 256 / TILE;

    impl Distribution<Acc2SrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Acc2SrcSoA {
            Acc2SrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
                rdot1: rng.gen(),
                rdot2: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc2 {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [Acc2SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [Acc2DstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [Acc2SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [Acc2DstSoA; NTILES] = [Default::default(); NTILES];
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