use super::{loop2, loop3, Compute, FromSoA, SplitAt, SplitAtMut, ToSoA};
use crate::{real::Real, sys::ParticleSystem};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3], [T; 3], [T; 3], [T; 3]);

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
pub struct SoaData {
    eps: [Real; THRESHOLD],
    mass: [Real; THRESHOLD],
    rdot: SoaDerivs<[Real; THRESHOLD]>,
    adot: SoaDerivs<[Real; THRESHOLD]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Derivs3(
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
    pub Vec<[Real; 3]>,
);
impl Derivs3 {
    pub fn zeros(n: usize) -> Self {
        Derivs3(
            vec![Default::default(); n],
            vec![Default::default(); n],
            vec![Default::default(); n],
            vec![Default::default(); n],
        )
    }

    pub fn as_slice(&self) -> Derivs3Slice<'_> {
        Derivs3Slice(&self.0[..], &self.1[..], &self.2[..], &self.3[..])
    }

    pub fn as_mut_slice(&mut self) -> Derivs3SliceMut<'_> {
        Derivs3SliceMut(
            &mut self.0[..],
            &mut self.1[..],
            &mut self.2[..],
            &mut self.3[..],
        )
    }
}

pub struct Derivs3Slice<'a>(
    pub &'a [[Real; 3]],
    pub &'a [[Real; 3]],
    pub &'a [[Real; 3]],
    pub &'a [[Real; 3]],
);
impl<'a, 'b: 'a> SplitAt<'a> for Derivs3Slice<'b> {
    type Output = Derivs3Slice<'a>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output) {
        let (lo0, hi0) = self.0.split_at(mid);
        let (lo1, hi1) = self.1.split_at(mid);
        let (lo2, hi2) = self.2.split_at(mid);
        let (lo3, hi3) = self.3.split_at(mid);
        (
            Derivs3Slice(lo0, lo1, lo2, lo3),
            Derivs3Slice(hi0, hi1, hi2, hi3),
        )
    }
}

pub struct Derivs3SliceMut<'a>(
    pub &'a mut [[Real; 3]],
    pub &'a mut [[Real; 3]],
    pub &'a mut [[Real; 3]],
    pub &'a mut [[Real; 3]],
);
impl<'a, 'b: 'a> SplitAtMut<'a> for Derivs3SliceMut<'b> {
    type Output = Derivs3SliceMut<'a>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn split_at_mut(&'a mut self, mid: usize) -> (Self::Output, Self::Output) {
        let (lo0, hi0) = self.0.split_at_mut(mid);
        let (lo1, hi1) = self.1.split_at_mut(mid);
        let (lo2, hi2) = self.2.split_at_mut(mid);
        let (lo3, hi3) = self.3.split_at_mut(mid);
        (
            Derivs3SliceMut(lo0, lo1, lo2, lo3),
            Derivs3SliceMut(hi0, hi1, hi2, hi3),
        )
    }
}
impl<'a> FromSoA for Derivs3SliceMut<'a> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            let minv = 1.0 / p.mass[j];
            for k in 0..3 {
                self.0[j][k] += p.adot.0[k][j] * minv;
                self.1[j][k] += p.adot.1[k][j] * minv;
                self.2[j][k] += p.adot.2[k][j] * minv;
                self.3[j][k] += p.adot.3[k][j] * minv;
            }
        }
    }
}

pub struct InpuData3Slice<'a> {
    pub eps: &'a [Real],
    pub mass: &'a [Real],
    pub rdot: Derivs3Slice<'a>,
}
impl<'a> InpuData3Slice<'a> {
    pub fn new(eps: &'a [Real], mass: &'a [Real], rdot: Derivs3Slice<'a>) -> Self {
        InpuData3Slice { eps, mass, rdot }
    }
}
impl<'a, 'b: 'a> SplitAt<'a> for InpuData3Slice<'b> {
    type Output = InpuData3Slice<'a>;

    fn len(&self) -> usize {
        self.rdot.len()
    }

    fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output) {
        let (eps_lo, eps_hi) = self.eps.split_at(mid);
        let (mass_lo, mass_hi) = self.mass.split_at(mid);
        let (rdot_lo, rdot_hi) = self.rdot.split_at(mid);
        (
            InpuData3Slice {
                eps: eps_lo,
                mass: mass_lo,
                rdot: rdot_lo,
            },
            InpuData3Slice {
                eps: eps_hi,
                mass: mass_hi,
                rdot: rdot_hi,
            },
        )
    }
}
impl<'a> ToSoA for InpuData3Slice<'a> {
    type SoaType = SoaData;

    fn to_soa(&self, p: &mut Self::SoaType) {
        let n = self.len();
        p.eps[..n].copy_from_slice(&self.eps[..n]);
        p.mass[..n].copy_from_slice(&self.mass[..n]);
        for j in 0..n {
            for k in 0..3 {
                p.rdot.0[k][j] = self.rdot.0[j][k];
                p.rdot.1[k][j] = self.rdot.1[j][k];
                p.rdot.2[k][j] = self.rdot.2[j][k];
                p.rdot.3[k][j] = self.rdot.3[j][k];
            }
        }
    }
}
impl<'a> From<&'a ParticleSystem> for InpuData3Slice<'a> {
    fn from(ps: &'a ParticleSystem) -> Self {
        InpuData3Slice {
            eps: &ps.attrs.eps[..],
            mass: &ps.attrs.mass[..],
            rdot: Derivs3Slice(
                &ps.attrs.pos[..],
                &ps.attrs.vel[..],
                &ps.attrs.acc0[..],
                &ps.attrs.acc1[..],
            ),
        }
    }
}

pub struct AccDot3Kernel {}
impl_kernel!(InpuData3Slice, Derivs3SliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for AccDot3Kernel {
    type Input = InpuData3Slice<'a>;
    type Output = Derivs3SliceMut<'a>;

    fn compute(&self, src: &Self::Input, dst: &mut Self::Output) {
        self.triangle(src, dst);
    }

    fn compute_mutual(
        &self,
        isrc: &Self::Input,
        jsrc: &Self::Input,
        idst: &mut Self::Output,
        jdst: &mut Self::Output,
    ) {
        self.rectangle(isrc, jsrc, idst, jdst);
    }
}

impl Kernel for AccDot3Kernel {
    // flop count: 157
    fn p2p(&self, ni: usize, nj: usize, ip: &mut SoaData, jp: &mut SoaData) {
        for ii in (0..ni).step_by(TILE) {
            for jj in (0..nj).step_by(TILE) {
                const CQ21: Real = 5.0 / 3.0;
                const CQ31: Real = 8.0 / 3.0;
                const CQ32: Real = 7.0 / 3.0;
                let mut drdot: SoaDerivs<[[Real; TILE]; TILE]> = Default::default();
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
                    drdot.0[k][i][j] = ip.rdot.0[k][ii + (j ^ i)] - jp.rdot.0[k][jj + j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.1[k][i][j] = ip.rdot.1[k][ii + (j ^ i)] - jp.rdot.1[k][jj + j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.2[k][i][j] = ip.rdot.2[k][ii + (j ^ i)] - jp.rdot.2[k][jj + j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.3[k][i][j] = ip.rdot.3[k][ii + (j ^ i)] - jp.rdot.3[k][jj + j];
                });

                loop2(TILE, TILE, |i, j| {
                    mm[i][j] = ip.mass[ii + (j ^ i)] * jp.mass[jj + j];
                });
                loop2(TILE, TILE, |i, j| {
                    s00[i][j] = ip.eps[ii + (j ^ i)] * jp.eps[jj + j];
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
                    ip.adot.0[k][ii + (j ^ i)] -= mm_r3[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp.adot.0[k][jj + j] += mm_r3[i][j] * drdot.0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip.adot.1[k][ii + (j ^ i)] -= mm_r3[i][j] * drdot.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp.adot.1[k][jj + j] += mm_r3[i][j] * drdot.1[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip.adot.2[k][ii + (j ^ i)] -= mm_r3[i][j] * drdot.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp.adot.2[k][jj + j] += mm_r3[i][j] * drdot.2[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ip.adot.3[k][ii + (j ^ i)] -= mm_r3[i][j] * drdot.3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp.adot.3[k][jj + j] += mm_r3[i][j] * drdot.3[k][i][j];
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

    const NTILES: usize = 256 / THRESHOLD;

    impl Distribution<SoaData> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SoaData {
            SoaData {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot: SoaDerivs(rng.gen(), rng.gen(), rng.gen(), rng.gen()),
                adot: Default::default(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = AccDot3Kernel {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ips: [SoaData; NTILES] = [Default::default(); NTILES];
            let mut jps: [SoaData; NTILES] = [Default::default(); NTILES];
            ips.iter_mut().for_each(|p| *p = rng.gen());
            jps.iter_mut().for_each(|p| *p = rng.gen());
            for i in 0..NTILES {
                for j in 0..NTILES {
                    kernel.p2p(THRESHOLD, THRESHOLD, &mut ips[i], &mut jps[j]);
                }
            }
            (ips, jps)
        });
    }
}

// -- end of file --
