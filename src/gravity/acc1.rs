use super::{loop2, loop3, Compute, FromSoA, SplitAt, SplitAtMut, ToSoA};
use crate::{real::Real, sys::ParticleSystem};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3], [T; 3]);

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
pub struct SoaData {
    eps: [Real; THRESHOLD],
    mass: [Real; THRESHOLD],
    rdot: SoaDerivs<[Real; THRESHOLD]>,
    adot: SoaDerivs<[Real; THRESHOLD]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Derivs1(pub Vec<[Real; 3]>, pub Vec<[Real; 3]>);
impl Derivs1 {
    pub fn zeros(n: usize) -> Self {
        Derivs1(vec![Default::default(); n], vec![Default::default(); n])
    }

    pub fn as_slice(&self) -> Derivs1Slice<'_> {
        Derivs1Slice(&self.0[..], &self.1[..])
    }

    pub fn as_mut_slice(&mut self) -> Derivs1SliceMut<'_> {
        Derivs1SliceMut(&mut self.0[..], &mut self.1[..])
    }
}

pub struct Derivs1Slice<'a>(pub &'a [[Real; 3]], pub &'a [[Real; 3]]);
impl<'a, 'b: 'a> SplitAt<'a> for Derivs1Slice<'b> {
    type Output = Derivs1Slice<'a>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output) {
        let (lo0, hi0) = self.0.split_at(mid);
        let (lo1, hi1) = self.1.split_at(mid);
        (Derivs1Slice(lo0, lo1), Derivs1Slice(hi0, hi1))
    }
}

pub struct Derivs1SliceMut<'a>(pub &'a mut [[Real; 3]], pub &'a mut [[Real; 3]]);
impl<'a, 'b: 'a> SplitAtMut<'a> for Derivs1SliceMut<'b> {
    type Output = Derivs1SliceMut<'a>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn split_at_mut(&'a mut self, mid: usize) -> (Self::Output, Self::Output) {
        let (lo0, hi0) = self.0.split_at_mut(mid);
        let (lo1, hi1) = self.1.split_at_mut(mid);
        (Derivs1SliceMut(lo0, lo1), Derivs1SliceMut(hi0, hi1))
    }
}
impl<'a> FromSoA for Derivs1SliceMut<'a> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            let minv = 1.0 / p.mass[j];
            for k in 0..3 {
                self.0[j][k] += p.adot.0[k][j] * minv;
                self.1[j][k] += p.adot.1[k][j] * minv;
            }
        }
    }
}

pub struct InpuData1Slice<'a> {
    pub eps: &'a [Real],
    pub mass: &'a [Real],
    pub rdot: Derivs1Slice<'a>,
}
impl<'a> InpuData1Slice<'a> {
    pub fn new(eps: &'a [Real], mass: &'a [Real], rdot: Derivs1Slice<'a>) -> Self {
        InpuData1Slice { eps, mass, rdot }
    }
}
impl<'a, 'b: 'a> SplitAt<'a> for InpuData1Slice<'b> {
    type Output = InpuData1Slice<'a>;

    fn len(&self) -> usize {
        self.rdot.len()
    }

    fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output) {
        let (eps_lo, eps_hi) = self.eps.split_at(mid);
        let (mass_lo, mass_hi) = self.mass.split_at(mid);
        let (rdot_lo, rdot_hi) = self.rdot.split_at(mid);
        (
            InpuData1Slice {
                eps: eps_lo,
                mass: mass_lo,
                rdot: rdot_lo,
            },
            InpuData1Slice {
                eps: eps_hi,
                mass: mass_hi,
                rdot: rdot_hi,
            },
        )
    }
}
impl<'a> ToSoA for InpuData1Slice<'a> {
    type SoaType = SoaData;

    fn to_soa(&self, p: &mut Self::SoaType) {
        let n = self.len();
        p.eps[..n].copy_from_slice(&self.eps[..n]);
        p.mass[..n].copy_from_slice(&self.mass[..n]);
        for j in 0..n {
            for k in 0..3 {
                p.rdot.0[k][j] = self.rdot.0[j][k];
                p.rdot.1[k][j] = self.rdot.1[j][k];
            }
        }
    }
}
impl<'a> From<&'a ParticleSystem> for InpuData1Slice<'a> {
    fn from(ps: &'a ParticleSystem) -> Self {
        InpuData1Slice {
            eps: &ps.attrs.eps[..],
            mass: &ps.attrs.mass[..],
            rdot: Derivs1Slice(&ps.attrs.pos[..], &ps.attrs.vel[..]),
        }
    }
}

pub struct AccDot1Kernel {}
impl_kernel!(InpuData1Slice, Derivs1SliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for AccDot1Kernel {
    type Input = InpuData1Slice<'a>;
    type Output = Derivs1SliceMut<'a>;

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

impl Kernel for AccDot1Kernel {
    // flop count: 56
    fn p2p(&self, ni: usize, nj: usize, ip: &mut SoaData, jp: &mut SoaData) {
        for ii in (0..ni).step_by(TILE) {
            for jj in (0..nj).step_by(TILE) {
                let mut drdot: SoaDerivs<[[Real; TILE]; TILE]> = Default::default();
                let mut s00: [[Real; TILE]; TILE] = Default::default();
                let mut s01: [[Real; TILE]; TILE] = Default::default();
                let mut q01: [[Real; TILE]; TILE] = Default::default();
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
                rdot: SoaDerivs(rng.gen(), rng.gen()),
                adot: Default::default(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = AccDot1Kernel {};
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
