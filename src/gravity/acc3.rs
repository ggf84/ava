use super::{loop2, loop3, Compute, FromSoA, ToSoA};
use crate::{
    sys::attributes::AttributesSlice,
    types::{AsSlice, AsSliceMut, Len, Real},
};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3], [T; 3], [T; 3], [T; 3]);

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
    pub Derivs3,
    pub Derivs3Slice,
    pub Derivs3SliceMut,
    {
        pub dot0: [Real; 3],
        pub dot1: [Real; 3],
        pub dot2: [Real; 3],
        pub dot3: [Real; 3],
    }
);

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub InpuData3,
    pub InpuData3Slice,
    pub InpuData3SliceMut,
    {
        pub eps: Real,
        pub mass: Real,
        pub rdot0: [Real; 3],
        pub rdot1: [Real; 3],
        pub rdot2: [Real; 3],
        pub rdot3: [Real; 3],
    }
);

impl FromSoA for Derivs3SliceMut<'_> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            let minv = 1.0 / p.mass[j];
            for k in 0..3 {
                self.dot0[j][k] += p.adot.0[k][j] * minv;
                self.dot1[j][k] += p.adot.1[k][j] * minv;
                self.dot2[j][k] += p.adot.2[k][j] * minv;
                self.dot3[j][k] += p.adot.3[k][j] * minv;
            }
        }
    }
}

impl ToSoA for InpuData3Slice<'_> {
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
                p.rdot.3[k][j] = self.rdot3[j][k];
            }
        }
    }
}

impl<'a, 'b: 'a> From<AttributesSlice<'b>> for InpuData3Slice<'a> {
    fn from(attrs: AttributesSlice<'b>) -> Self {
        InpuData3Slice {
            _len: attrs.len(),
            eps: &attrs.eps[..],
            mass: &attrs.mass[..],
            rdot0: &attrs.pos[..],
            rdot1: &attrs.vel[..],
            rdot2: &attrs.acc0[..],
            rdot3: &attrs.acc1[..],
        }
    }
}

pub struct AccDot3Kernel {}
impl_kernel!(InpuData3Slice, Derivs3SliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for AccDot3Kernel {
    type Input = InpuData3Slice<'a>;
    type Output = Derivs3SliceMut<'a>;

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
