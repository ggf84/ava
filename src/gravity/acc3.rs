use super::{loop1, loop2, loop3, Compute, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::ParticleSystem};
use soa_derive::StructOfArray;

#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct Derivs<T>([T; 3], [T; 3], [T; 3], [T; 3]);

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
    rdot1: [Real; 3],
    rdot2: [Real; 3],
    rdot3: [Real; 3],
}

#[derive(Clone, Default, Debug, PartialEq, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq"]
struct Dst {
    adot0: [Real; 3],
    adot1: [Real; 3],
    adot2: [Real; 3],
    adot3: [Real; 3],
}

impl<'a> ToSoA<SrcSoA> for SrcSlice<'a> {
    fn to_soa(&self, ps_src: &mut [SrcSoA]) {
        let n = self.len();
        let mut jj = 0;
        for p_src in ps_src.iter_mut() {
            for j in 0..TILE {
                if jj < n {
                    p_src.eps[j] = self.eps[jj];
                    p_src.mass[j] = self.mass[jj];
                    loop1(3, |k| p_src.rdot.0[k][j] = self.rdot0[jj][k]);
                    loop1(3, |k| p_src.rdot.1[k][j] = self.rdot1[jj][k]);
                    loop1(3, |k| p_src.rdot.2[k][j] = self.rdot2[jj][k]);
                    loop1(3, |k| p_src.rdot.3[k][j] = self.rdot3[jj][k]);
                    jj += 1;
                }
            }
        }
    }
}

impl<'a> FromSoA<SrcSoA, DstSoA> for DstSliceMut<'a> {
    fn from_soa(&mut self, ps_src: &[SrcSoA], ps_dst: &[DstSoA]) {
        let n = self.len();
        let mut jj = 0;
        for (p_src, p_dst) in ps_src.iter().zip(ps_dst.iter()) {
            for j in 0..TILE {
                if jj < n {
                    let minv = 1.0 / p_src.mass[j];
                    loop1(3, |k| self.adot0[jj][k] += p_dst.adot.0[k][j] * minv);
                    loop1(3, |k| self.adot1[jj][k] += p_dst.adot.1[k][j] * minv);
                    loop1(3, |k| self.adot2[jj][k] += p_dst.adot.2[k][j] * minv);
                    loop1(3, |k| self.adot3[jj][k] += p_dst.adot.3[k][j] * minv);
                    jj += 1;
                }
            }
        }
    }
}

pub struct AccDot3Kernel {}
impl_kernel!(SrcSlice, DstSliceMut, SrcSoA, DstSoA, 64);

#[derive(Clone, Debug, PartialEq)]
pub struct AccDot3(
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
);
impl AccDot3 {
    pub fn zeros(n: usize) -> Self {
        AccDot3(
            vec![Default::default(); n],
            vec![Default::default(); n],
            vec![Default::default(); n],
            vec![Default::default(); n],
        )
    }
}
impl<'a> From<&'a mut AccDot3> for DstSliceMut<'a> {
    fn from(acc: &'a mut AccDot3) -> Self {
        DstSliceMut {
            adot0: &mut acc.0[..],
            adot1: &mut acc.1[..],
            adot2: &mut acc.2[..],
            adot3: &mut acc.3[..],
        }
    }
}

impl<'a, T> From<&'a T> for SrcSlice<'a>
where
    T: AsRef<ParticleSystem>,
{
    fn from(ps: &'a T) -> Self {
        let ps = ps.as_ref();
        SrcSlice {
            eps: &ps.attrs.eps[..],
            mass: &ps.attrs.mass[..],
            rdot0: &ps.attrs.pos[..],
            rdot1: &ps.attrs.vel[..],
            rdot2: &ps.attrs.acc0[..],
            rdot3: &ps.attrs.acc1[..],
        }
    }
}

impl<T: Into<SrcSlice<'_>>> Compute<T> for AccDot3Kernel {
    type Output = AccDot3;
    fn compute(&self, src: T, dst: &mut Self::Output) {
        let src = src.into();
        let mut dst = dst.into();
        self.triangle(&src, &mut dst);
    }
    fn compute_mutual(&self, isrc: T, jsrc: T, idst: &mut Self::Output, jdst: &mut Self::Output) {
        let isrc = isrc.into();
        let jsrc = jsrc.into();
        let mut idst = idst.into();
        let mut jdst = jdst.into();
        self.rectangle(&isrc, &mut idst, &jsrc, &mut jdst);
    }
}

impl Kernel for AccDot3Kernel {
    // flop count: 157
    fn p2p(
        &self,
        ip_src: &[SrcSoA],
        ip_dst: &mut [DstSoA],
        jp_src: &[SrcSoA],
        jp_dst: &mut [DstSoA],
    ) {
        for (ip_src, ip_dst) in ip_src.iter().zip(ip_dst.iter_mut()) {
            for (jp_src, jp_dst) in jp_src.iter().zip(jp_dst.iter_mut()) {
                const CQ21: Real = 5.0 / 3.0;
                const CQ31: Real = 8.0 / 3.0;
                const CQ32: Real = 7.0 / 3.0;
                let mut drdot: Derivs<[[Real; TILE]; TILE]> = Default::default();
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
                    drdot.0[k][i][j] = ip_src.rdot.0[k][j ^ i] - jp_src.rdot.0[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.1[k][i][j] = ip_src.rdot.1[k][j ^ i] - jp_src.rdot.1[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.2[k][i][j] = ip_src.rdot.2[k][j ^ i] - jp_src.rdot.2[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.3[k][i][j] = ip_src.rdot.3[k][j ^ i] - jp_src.rdot.3[k][j];
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
                loop3(3, TILE, TILE, |k, i, j| {
                    s01[i][j] += drdot.0[k][i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s11[i][j] += drdot.1[k][i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s02[i][j] += drdot.0[k][i][j] * drdot.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s12[i][j] += drdot.1[k][i][j] * drdot.2[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s02[i][j] += s11[i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s03[i][j] += drdot.0[k][i][j] * drdot.3[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s03[i][j] += 3.0 * s12[i][j];
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
                    q23[i][j] = 3.0 * q01[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    q13[i][j] = 3.0 * q02[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    q12[i][j] = 2.0 * q01[i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.3[k][i][j] -= q23[i][j] * drdot.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.3[k][i][j] -= q13[i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.3[k][i][j] -= q03[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.2[k][i][j] -= q12[i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.2[k][i][j] -= q02[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.1[k][i][j] -= q01[i][j] * drdot.0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip_dst.adot.0[k][j ^ i] -= mm_r3[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp_dst.adot.0[k][j] += mm_r3[i][j] * drdot.0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip_dst.adot.1[k][j ^ i] -= mm_r3[i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp_dst.adot.1[k][j] += mm_r3[i][j] * drdot.1[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip_dst.adot.2[k][j ^ i] -= mm_r3[i][j] * drdot.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp_dst.adot.2[k][j] += mm_r3[i][j] * drdot.2[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip_dst.adot.3[k][j ^ i] -= mm_r3[i][j] * drdot.3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp_dst.adot.3[k][j] += mm_r3[i][j] * drdot.3[k][i][j];
                });
            }
        }
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
                rdot: Derivs(rng.gen(), rng.gen(), rng.gen(), rng.gen()),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = AccDot3Kernel {};
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
