use super::{loop2, loop3, Compute, FromSoA, ToSoA};
use crate::{
    sys::attributes::AttributesSlice,
    types::{AsSlice, AsSliceMut, Len, Real},
};

const THRESHOLD: usize = 32;
const TILE: usize = 16 / std::mem::size_of::<Real>();

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SoaDerivs<T>([T; 3], [T; 3]);

#[repr(align(16))]
#[derive(Copy, Clone, Default)]
pub(super) struct SoaData {
    eps: [Real; THRESHOLD],
    mass: [Real; THRESHOLD],
    rdot: SoaDerivs<[Real; THRESHOLD]>,
    ekin: [Real; THRESHOLD],
    epot: [Real; THRESHOLD],
}

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub Energy,
    pub EnergySlice,
    pub EnergySliceMut,
    {
        pub kin: Real,
        pub pot: Real,
    }
);

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    pub InpuData1,
    pub InpuData1Slice,
    pub InpuData1SliceMut,
    {
        pub eps: Real,
        pub mass: Real,
        pub rdot0: [Real; 3],
        pub rdot1: [Real; 3],
    }
);

impl Energy {
    pub fn reduce(&self, mtot: Real) -> (Real, Real) {
        let ke = 0.25 * self.kin.iter().sum::<Real>();
        let pe = 0.50 * self.pot.iter().sum::<Real>();

        (ke / mtot, pe)
    }
}

impl FromSoA for EnergySliceMut<'_> {
    type SoaType = SoaData;

    fn from_soa(&mut self, p: &Self::SoaType) {
        let n = self.len();
        for j in 0..n {
            self.kin[j] += p.ekin[j];
            self.pot[j] += p.epot[j];
        }
    }
}

impl ToSoA for InpuData1Slice<'_> {
    type SoaType = SoaData;

    fn to_soa(&self, p: &mut Self::SoaType) {
        let n = self.len();
        p.eps[..n].copy_from_slice(&self.eps[..n]);
        p.mass[..n].copy_from_slice(&self.mass[..n]);
        for j in 0..n {
            for k in 0..3 {
                p.rdot.0[k][j] = self.rdot0[j][k];
                p.rdot.1[k][j] = self.rdot1[j][k];
            }
        }
    }
}

impl<'a, 'b: 'a> From<AttributesSlice<'b>> for InpuData1Slice<'a> {
    fn from(attrs: AttributesSlice<'b>) -> Self {
        InpuData1Slice {
            _len: attrs.len(),
            eps: &attrs.eps[..],
            mass: &attrs.mass[..],
            rdot0: &attrs.pos[..],
            rdot1: &attrs.vel[..],
        }
    }
}

/// Compute the kinetic and potential energies of the system.
///
/// \\[ KE = \frac{1}{4 M} \sum_{i=0}^{N} \sum_{j=0}^{N} m_{i} m_{j} v_{ij}^{2} \\]
///
/// \\[ PE = -\frac{1}{2} \sum_{i=0}^{N} \sum_{j=0}^{N} \frac{m_{i} m_{j}}{r_{ij}} \\]
///
/// where \\( M \\) is the total mass, and \\( N \\) is the number of particles in the system.
///
/// Thus, \\[ KE_{tot} = KE + KE_{CoM}\\]
///
/// and, \\[ PE_{tot} = PE \\]
///
/// Compute the mutual kinetic and potential energies of two disjoint systems, A and B.
///
/// \\[ KE_{AB} = \frac{1}{4 M} \sum_{i=0}^{N_{A}} \sum_{j=0}^{N_{B}} m_{i} m_{j} v_{ij}^{2} \\]
///
/// \\[ PE_{AB} = -\frac{1}{2} \sum_{i=0}^{N_{A}} \sum_{j=0}^{N_{B}} \frac{m_{i} m_{j}}{r_{ij}} \\]
///
/// where \\( M = M_{A} + M_{B} \\) is the total mass of the combined system, and \\( N_{A} \\)
/// and \\( N_{B} \\) are the number of particles in each system.
///
/// Thus, \\[ KE_{tot} = \frac{M_{A} KE_{A} + M_{B} KE_{B}}{M} + KE_{AB} + KE_{CoM} \\]
///
/// and, \\[ PE_{tot} = PE_{A} + PE_{B} + PE_{AB} \\]
///
pub struct EnergyKernel {}
impl_kernel!(InpuData1Slice, EnergySliceMut, SoaData, THRESHOLD);

impl<'a> Compute<'a> for EnergyKernel {
    type Input = InpuData1Slice<'a>;
    type Output = EnergySliceMut<'a>;

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

impl Kernel for EnergyKernel {
    // flop count: 28
    fn p2p(&self, ni: usize, nj: usize, ip: &mut SoaData, jp: &mut SoaData) {
        for ii in (0..ni).step_by(TILE) {
            for jj in (0..nj).step_by(TILE) {
                let mut drdot: SoaDerivs<[[Real; TILE]; TILE]> = Default::default();
                let mut s00: [[Real; TILE]; TILE] = Default::default();
                let mut s11: [[Real; TILE]; TILE] = Default::default();
                let mut rinv1: [[Real; TILE]; TILE] = Default::default();
                let mut rinv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm: [[Real; TILE]; TILE] = Default::default();
                let mut mmv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm_r1: [[Real; TILE]; TILE] = Default::default();

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
                    s11[i][j] += drdot.1[k][i][j] * drdot.1[k][i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    mmv2[i][j] = mm[i][j] * s11[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] = s00[i][j].recip();
                });
                loop2(TILE, TILE, |i, j| {
                    rinv1[i][j] = rinv2[i][j].sqrt();
                });
                loop2(TILE, TILE, |i, j| {
                    mm_r1[i][j] = mm[i][j] * rinv1[i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    ip.ekin[ii + (j ^ i)] += mmv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    jp.ekin[jj + j] += mmv2[i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    ip.epot[ii + (j ^ i)] -= mm_r1[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    jp.epot[jj + j] -= mm_r1[i][j];
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
                ekin: Default::default(),
                epot: Default::default(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = EnergyKernel {};
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
