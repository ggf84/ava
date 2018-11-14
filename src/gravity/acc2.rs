use super::{loop2, loop3, Compute, FromSoA, ToSoA};
use crate::{
    sys::attributes::AttributesSlice,
    types::{AsSlice, AsSliceMut, Len, Real},
};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3], [T; 3], [T; 3]);

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
pub(super) struct SoaData {
    eps: [Real; THRESHOLD],
    mass: [Real; THRESHOLD],
    rdot: SoaDerivs<[Real; THRESHOLD]>,
    adot: SoaDerivs<[Real; THRESHOLD]>,
}

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub Derivs2,
    pub Derivs2Slice,
    pub Derivs2SliceMut,
    {
        pub dot0: [Real; 3],
        pub dot1: [Real; 3],
        pub dot2: [Real; 3],
    }
);

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub InpuData2,
    pub InpuData2Slice,
    pub InpuData2SliceMut,
    {
        pub eps: Real,
        pub mass: Real,
        pub rdot0: [Real; 3],
        pub rdot1: [Real; 3],
        pub rdot2: [Real; 3],
    }
);

impl FromSoA for Derivs2SliceMut<'_> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            let minv = 1.0 / p.mass[j];
            for k in 0..3 {
                self.dot0[j][k] += p.adot.0[k][j] * minv;
                self.dot1[j][k] += p.adot.1[k][j] * minv;
                self.dot2[j][k] += p.adot.2[k][j] * minv;
            }
        }
    }
}

impl ToSoA for InpuData2Slice<'_> {
    type SoaType = SoaData;

    fn to_soa(&self, p: &mut Self::SoaType) {
        let n = self.len();
        p.eps[..n].copy_from_slice(&self.eps[..n]);
        p.mass[..n].copy_from_slice(&self.mass[..n]);
        for j in 0..n {
            for k in 0..3 {
                p.rdot.0[k][j] = self.rdot0[j][k];
                p.rdot.1[k][j] = self.rdot1[j][k];
                p.rdot.2[k][j] = self.rdot2[j][k];
            }
        }
    }
}

impl<'a, 'b: 'a> From<AttributesSlice<'b>> for InpuData2Slice<'a> {
    fn from(attrs: AttributesSlice<'b>) -> Self {
        InpuData2Slice {
            _len: attrs.len(),
            eps: &attrs.eps[..],
            mass: &attrs.mass[..],
            rdot0: &attrs.pos[..],
            rdot1: &attrs.vel[..],
            rdot2: &attrs.acc0[..],
        }
    }
}

pub struct AccDot2Kernel {}
impl_kernel!(InpuData2Slice, Derivs2SliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for AccDot2Kernel {
    type Input = InpuData2Slice<'a>;
    type Output = Derivs2SliceMut<'a>;

    fn compute(&self, src: Self::Input, dst: Self::Output) {
        self.triangle(src, dst);
    }

    fn compute_mutual(
        &self,
        isrc: Self::Input,
        jsrc: Self::Input,
        idst: Self::Output,
        jdst: Self::Output,
    ) {
        self.rectangle(isrc, jsrc, idst, jdst);
    }
}

impl Kernel for AccDot2Kernel {
    // flop count: 101
    fn p2p(&self, ni: usize, nj: usize, ip: &mut SoaData, jp: &mut SoaData) {
        for ii in (0..ni).step_by(TILE) {
            for jj in (0..nj).step_by(TILE) {
                const CQ21: Real = 5.0 / 3.0;
                let mut drdot: SoaDerivs<[[Real; TILE]; TILE]> = Default::default();
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
                    drdot.0[k][i][j] = ip.rdot.0[k][ii + (j ^ i)] - jp.rdot.0[k][jj + j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.1[k][i][j] = ip.rdot.1[k][ii + (j ^ i)] - jp.rdot.1[k][jj + j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.2[k][i][j] = ip.rdot.2[k][ii + (j ^ i)] - jp.rdot.2[k][jj + j];
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
                rdot: SoaDerivs(rng.gen(), rng.gen(), rng.gen()),
                adot: Default::default(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = AccDot2Kernel {};
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
