use super::{loop2, loop3, Compute, FromSoA, ToSoA};
use crate::{
    sys::attributes::AttributesSlice,
    types::{AsSlice, AsSliceMut, Len, Real},
};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3]);

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
    pub Derivs0,
    pub Derivs0Slice,
    pub Derivs0SliceMut,
    {
        pub dot0: [Real; 3],
    }
);

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub InpuData0,
    pub InpuData0Slice,
    pub InpuData0SliceMut,
    {
        pub eps: Real,
        pub mass: Real,
        pub rdot0: [Real; 3],
    }
);

impl FromSoA for Derivs0SliceMut<'_> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            let minv = 1.0 / p.mass[j];
            for k in 0..3 {
                self.dot0[j][k] += p.adot.0[k][j] * minv;
            }
        }
    }
}

impl ToSoA for InpuData0Slice<'_> {
    type SoaType = SoaData;

    fn to_soa(&self, p: &mut Self::SoaType) {
        let n = self.len();
        p.eps[..n].copy_from_slice(&self.eps[..n]);
        p.mass[..n].copy_from_slice(&self.mass[..n]);
        for j in 0..n {
            for k in 0..3 {
                p.rdot.0[k][j] = self.rdot0[j][k];
            }
        }
    }
}

impl<'a, 'b: 'a> From<AttributesSlice<'b>> for InpuData0Slice<'a> {
    fn from(attrs: AttributesSlice<'b>) -> Self {
        InpuData0Slice {
            _len: attrs.len(),
            eps: &attrs.eps[..],
            mass: &attrs.mass[..],
            rdot0: &attrs.pos[..],
        }
    }
}

pub struct AccDot0Kernel {}
impl_kernel!(InpuData0Slice, Derivs0SliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for AccDot0Kernel {
    type Input = InpuData0Slice<'a>;
    type Output = Derivs0SliceMut<'a>;

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

impl Kernel for AccDot0Kernel {
    // flop count: 27
    fn p2p(&self, ni: usize, nj: usize, ip: &mut SoaData, jp: &mut SoaData) {
        for ii in (0..ni).step_by(TILE) {
            for jj in (0..nj).step_by(TILE) {
                let mut drdot: SoaDerivs<[[Real; TILE]; TILE]> = Default::default();
                let mut s00: [[Real; TILE]; TILE] = Default::default();
                let mut rinv1: [[Real; TILE]; TILE] = Default::default();
                let mut rinv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm: [[Real; TILE]; TILE] = Default::default();
                let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.0[k][i][j] = ip.rdot.0[k][ii + (j ^ i)] - jp.rdot.0[k][jj + j];
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
                    ip.adot.0[k][ii + (j ^ i)] -= mm_r3[i][j] * drdot.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    jp.adot.0[k][jj + j] += mm_r3[i][j] * drdot.0[k][i][j];
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
        rngs::SmallRng,
        Rng, SeedableRng,
    };
    use test::Bencher;

    const NTILES: usize = 256 / THRESHOLD;

    impl Distribution<SoaData> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SoaData {
            SoaData {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot: SoaDerivs(rng.gen()),
                adot: Default::default(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = AccDot0Kernel {};
        let mut rng = SmallRng::seed_from_u64(1234567890);
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
