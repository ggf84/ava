use super::{loop1, loop2, loop3, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::Particle};
use soa_derive::StructOfArray;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct Acc0SrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct Acc0DstSoA {
    adot0: [[Real; TILE]; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct Acc0Src {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct Acc0Dst {
    adot0: [Real; 3],
}

impl<'a> ToSoA for Acc0SrcSlice<'a> {
    type SrcTypeSoA = Acc0SrcSoA;
    fn to_soa(&self, ps_src: &mut [Self::SrcTypeSoA]) {
        let n = self.len();
        let mut jj = 0;
        for p_src in ps_src.iter_mut() {
            for j in 0..TILE {
                if jj < n {
                    p_src.eps[j] = self.eps[jj];
                    p_src.mass[j] = self.mass[jj];
                    loop1(3, |k| p_src.rdot0[k][j] = self.rdot0[jj][k]);
                    jj += 1;
                }
            }
        }
    }
}

impl<'a> FromSoA for Acc0DstSliceMut<'a> {
    type SrcTypeSoA = Acc0SrcSoA;
    type DstTypeSoA = Acc0DstSoA;
    fn from_soa(&mut self, ps_src: &[Self::SrcTypeSoA], ps_dst: &[Self::DstTypeSoA]) {
        let n = self.len();
        let mut jj = 0;
        for (p_src, p_dst) in ps_src.iter().zip(ps_dst.iter()) {
            for j in 0..TILE {
                if jj < n {
                    let minv = 1.0 / p_src.mass[j];
                    loop1(3, |k| self.adot0[jj][k] += p_dst.adot0[k][j] * minv);
                    jj += 1;
                }
            }
        }
    }
}

pub struct Acc0 {}
impl_kernel!(Acc0SrcSlice, Acc0DstSliceMut, Acc0SrcSoA, Acc0DstSoA, 64);

impl Kernel for Acc0 {
    // flop count: 27
    fn p2p(
        &self,
        ip_src: &Acc0SrcSoA,
        ip_dst: &mut Acc0DstSoA,
        jp_src: &Acc0SrcSoA,
        jp_dst: &mut Acc0DstSoA,
    ) {
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

        loop3(3, TILE, TILE, |k, i, j| {
            drdot0[k][i][j] = ip_src.rdot0[k][j ^ i] - jp_src.rdot0[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += drdot0[k][i][j] * drdot0[k][i][j];
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

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot0[k][j ^ i] -= mm_r3[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot0[k][j] += mm_r3[i][j] * drdot0[k][i][j];
        });
    }
}

impl Acc0 {
    /// For each particle of the system, compute the {0}-derivative of the gravitational acceleration.
    pub fn compute(&self, psys: &[Particle]) -> (Vec<[Real; 3]>,) {
        let mut src = Acc0SrcVec::with_capacity(psys.len());
        let mut dst = Acc0DstVec::with_capacity(psys.len());
        for p in psys.iter() {
            src.push(Acc0Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
            });
            dst.push(Default::default());
        }

        self.triangle(&src.as_slice(), &mut dst.as_mut_slice());

        (dst.adot0,)
    }
    /// For each particle of two disjoint systems, compute the mutual {0}-derivative of the gravitational acceleration.
    pub fn compute_mutual(
        &self,
        ipsys: &[Particle],
        jpsys: &[Particle],
    ) -> ((Vec<[Real; 3]>,), (Vec<[Real; 3]>,)) {
        let mut isrc = Acc0SrcVec::with_capacity(ipsys.len());
        let mut idst = Acc0DstVec::with_capacity(ipsys.len());
        for p in ipsys.iter() {
            isrc.push(Acc0Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
            });
            idst.push(Default::default());
        }

        let mut jsrc = Acc0SrcVec::with_capacity(jpsys.len());
        let mut jdst = Acc0DstVec::with_capacity(jpsys.len());
        for p in jpsys.iter() {
            jsrc.push(Acc0Src {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
            });
            jdst.push(Default::default());
        }

        self.rectangle(
            &isrc.as_slice(),
            &mut idst.as_mut_slice(),
            &jsrc.as_slice(),
            &mut jdst.as_mut_slice(),
        );

        ((idst.adot0,), (jdst.adot0,))
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

    impl Distribution<Acc0SrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Acc0SrcSoA {
            Acc0SrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc0 {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [Acc0SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [Acc0DstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [Acc0SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [Acc0DstSoA; NTILES] = [Default::default(); NTILES];
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
