use super::{loop1, loop2, loop3, Compute, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::ParticleSystem};
use soa_derive::StructOfArray;

#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct Derivs<T>([T; 3]);

#[repr(align(16))]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct SrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot: Derivs<[Real; TILE]>,
}

#[repr(align(16))]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct DstSoA {
    adot: Derivs<[Real; TILE]>,
}

#[derive(Clone, Default, Debug, PartialEq, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq"]
pub struct Src {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
}

#[derive(Clone, Default, Debug, PartialEq, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq"]
pub struct Dst {
    adot0: [Real; 3],
}

impl<'a> ToSoA<[SrcSoA]> for SrcSlice<'a> {
    fn to_soa(&self, ps_src: &mut [SrcSoA]) {
        let n = self.len();
        let mut jj = 0;
        for p_src in ps_src.iter_mut() {
            for j in 0..TILE {
                if jj < n {
                    p_src.eps[j] = self.eps[jj];
                    p_src.mass[j] = self.mass[jj];
                    loop1(3, |k| p_src.rdot.0[k][j] = self.rdot0[jj][k]);
                    jj += 1;
                }
            }
        }
    }
}

impl<'a> FromSoA<[SrcSoA], [DstSoA]> for DstSliceMut<'a> {
    fn from_soa(&mut self, ps_src: &[SrcSoA], ps_dst: &[DstSoA]) {
        let n = self.len();
        let mut jj = 0;
        for (p_src, p_dst) in ps_src.iter().zip(ps_dst.iter()) {
            for j in 0..TILE {
                if jj < n {
                    let minv = 1.0 / p_src.mass[j];
                    loop1(3, |k| self.adot0[jj][k] += p_dst.adot.0[k][j] * minv);
                    jj += 1;
                }
            }
        }
    }
}

pub struct Acc0 {}
impl_kernel!(SrcSlice, DstSliceMut, SrcSoA, DstSoA, 64);

impl Kernel for Acc0 {
    // flop count: 27
    fn p2p(
        &self,
        ip_src: &[SrcSoA],
        ip_dst: &mut [DstSoA],
        jp_src: &[SrcSoA],
        jp_dst: &mut [DstSoA],
    ) {
        for (ip_src, ip_dst) in ip_src.iter().zip(ip_dst.iter_mut()) {
            for (jp_src, jp_dst) in jp_src.iter().zip(jp_dst.iter_mut()) {
                let mut drdot: Derivs<[[Real; TILE]; TILE]> = Default::default();
                let mut s00: [[Real; TILE]; TILE] = Default::default();
                let mut rinv1: [[Real; TILE]; TILE] = Default::default();
                let mut rinv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm: [[Real; TILE]; TILE] = Default::default();
                let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.0[k][i][j] = ip_src.rdot.0[k][j ^ i] - jp_src.rdot.0[k][j];
                });

                loop2(TILE, TILE, |i, j| {
                    mm[i][j] = ip_src.mass[j ^ i] * jp_src.mass[j];
                });
                loop2(TILE, TILE, |i, j| {
                    s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    s00[i][j] += drdot.0[k][i][j] * drdot.0[k][i][j];
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
                    ip_dst.adot.0[k][j ^ i] -= mm_r3[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp_dst.adot.0[k][j] += mm_r3[i][j] * drdot.0[k][i][j];
                });
            }
        }
    }
}

impl<'a, 'b: 'a> From<&'b ParticleSystem> for SrcSlice<'a> {
    fn from(ps: &'b ParticleSystem) -> Self {
        SrcSlice {
            eps: &ps.attrs.eps[..],
            mass: &ps.attrs.mass[..],
            rdot0: &ps.attrs.pos[..],
        }
    }
}

impl<S: Into<SrcSlice<'_>>> Compute<S> for Acc0 {
    type Output = (Vec<[Real; 3]>,);
    /// For each particle of the system, compute the k-th derivative
    /// of the gravitational acceleration, for k = {0}.
    fn compute(&self, src: S) -> Self::Output {
        let src = src.into();
        let mut acc = (vec![Default::default(); src.len()],);
        let mut dst = DstSliceMut {
            adot0: &mut acc.0[..],
        };

        self.triangle(&src, &mut dst);
        acc
    }
    /// For each particle of two disjoint systems, A and B, compute the mutual k-th derivative
    /// of the gravitational acceleration, for k = {0}.
    fn compute_mutual(&self, isrc: S, jsrc: S) -> (Self::Output, Self::Output) {
        let isrc = isrc.into();
        let jsrc = jsrc.into();
        let mut iacc = (vec![Default::default(); isrc.len()],);
        let mut jacc = (vec![Default::default(); jsrc.len()],);
        let mut idst = DstSliceMut {
            adot0: &mut iacc.0[..],
        };
        let mut jdst = DstSliceMut {
            adot0: &mut jacc.0[..],
        };

        self.rectangle(&isrc, &mut idst, &jsrc, &mut jdst);
        (iacc, jacc)
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

    impl Distribution<SrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SrcSoA {
            SrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot: Derivs(rng.gen()),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc0 {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [DstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [SrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [DstSoA; NTILES] = [Default::default(); NTILES];
            ip_src.iter_mut().for_each(|p| *p = rng.gen());
            jp_src.iter_mut().for_each(|p| *p = rng.gen());
            kernel.p2p(
                &ip_src[..NTILES],
                &mut ip_dst[..NTILES],
                &jp_src[..NTILES],
                &mut jp_dst[..NTILES],
            );
            (ip_dst, jp_dst)
        });
    }
}

// -- end of file --
