use super::{loop1, loop2, loop3, Compute, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::Particle};
use soa_derive::StructOfArray;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct EnergySrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
    rdot1: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct EnergyDstSoA {
    ekin: [Real; TILE],
    epot: [Real; TILE],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct EnergySrc {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
    rdot1: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct EnergyDst {
    ekin: Real,
    epot: Real,
}

impl<'a> ToSoA<[EnergySrcSoA]> for EnergySrcSlice<'a> {
    fn to_soa(&self, ps_src: &mut [EnergySrcSoA]) {
        let n = self.len();
        let mut jj = 0;
        for p_src in ps_src.iter_mut() {
            for j in 0..TILE {
                if jj < n {
                    p_src.eps[j] = self.eps[jj];
                    p_src.mass[j] = self.mass[jj];
                    loop1(3, |k| p_src.rdot0[k][j] = self.rdot0[jj][k]);
                    loop1(3, |k| p_src.rdot1[k][j] = self.rdot1[jj][k]);
                    jj += 1;
                }
            }
        }
    }
}

impl<'a> FromSoA<[EnergySrcSoA], [EnergyDstSoA]> for EnergyDstSliceMut<'a> {
    fn from_soa(&mut self, ps_src: &[EnergySrcSoA], ps_dst: &[EnergyDstSoA]) {
        let n = self.len();
        let mut jj = 0;
        for (_p_src, p_dst) in ps_src.iter().zip(ps_dst.iter()) {
            for j in 0..TILE {
                if jj < n {
                    self.ekin[jj] += p_dst.ekin[j];
                    self.epot[jj] += p_dst.epot[j];
                    jj += 1;
                }
            }
        }
    }
}

pub struct Energy {
    mtot: Real,
}
impl_kernel!(
    EnergySrcSlice,
    EnergyDstSliceMut,
    EnergySrcSoA,
    EnergyDstSoA,
    64,
);

impl Kernel for Energy {
    // flop count: 28
    fn p2p(
        &self,
        ip_src: &EnergySrcSoA,
        ip_dst: &mut EnergyDstSoA,
        jp_src: &EnergySrcSoA,
        jp_dst: &mut EnergyDstSoA,
    ) {
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s11: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mmv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r1: [[Real; TILE]; TILE] = Default::default();

        loop3(3, TILE, TILE, |k, i, j| {
            drdot0[k][i][j] = ip_src.rdot0[k][j ^ i] - jp_src.rdot0[k][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            drdot1[k][i][j] = ip_src.rdot1[k][j ^ i] - jp_src.rdot1[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += drdot0[k][i][j] * drdot0[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] = s00[i][j].recip();
        });
        loop2(TILE, TILE, |i, j| {
            rinv1[i][j] = rinv2[i][j].sqrt();
        });

        loop2(TILE, TILE, |i, j| {
            s11[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s11[i][j] += drdot1[k][i][j] * drdot1[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip_src.mass[j ^ i] * jp_src.mass[j];
        });
        loop2(TILE, TILE, |i, j| {
            mmv2[i][j] = mm[i][j] * s11[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            mm_r1[i][j] = mm[i][j] * rinv1[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            ip_dst.ekin[j ^ i] += mmv2[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            jp_dst.ekin[j] += mmv2[i][j];
        });

        loop2(TILE, TILE, |i, j| {
            ip_dst.epot[j ^ i] -= mm_r1[i][j];
        });
        loop2(TILE, TILE, |i, j| {
            jp_dst.epot[j] -= mm_r1[i][j];
        });
    }
}

impl Compute<[Particle]> for Energy {
    type Output = (Vec<Real>, Vec<Real>);
    fn compute(&self, psys: &[Particle]) -> Self::Output {
        let mut src = EnergySrcVec::with_capacity(psys.len());
        let mut dst = EnergyDstVec::with_capacity(psys.len());
        for p in psys.iter() {
            src.push(EnergySrc {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
            });
            dst.push(Default::default());
        }

        self.triangle(&src.as_slice(), &mut dst.as_mut_slice());

        (dst.ekin, dst.epot)
    }
    fn compute_mutual(
        &self,
        ipsys: &[Particle],
        jpsys: &[Particle],
    ) -> (Self::Output, Self::Output) {
        let mut isrc = EnergySrcVec::with_capacity(ipsys.len());
        let mut idst = EnergyDstVec::with_capacity(ipsys.len());
        for p in ipsys.iter() {
            isrc.push(EnergySrc {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
            });
            idst.push(Default::default());
        }

        let mut jsrc = EnergySrcVec::with_capacity(jpsys.len());
        let mut jdst = EnergyDstVec::with_capacity(jpsys.len());
        for p in jpsys.iter() {
            jsrc.push(EnergySrc {
                eps: p.eps,
                mass: p.mass,
                rdot0: p.pos,
                rdot1: p.vel,
            });
            jdst.push(Default::default());
        }

        self.rectangle(
            &isrc.as_slice(),
            &mut idst.as_mut_slice(),
            &jsrc.as_slice(),
            &mut jdst.as_mut_slice(),
        );

        ((idst.ekin, idst.epot), (jdst.ekin, jdst.epot))
    }
}

impl Energy {
    pub fn new(mtot: Real) -> Self {
        Energy { mtot }
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
    pub fn energies(&self, psys: &[Particle]) -> (Real, Real) {
        let (ke, pe) = self.compute(psys);

        let pe = pe.iter().sum::<Real>() * 0.5;
        let ke = ke.iter().sum::<Real>() * 0.25 / self.mtot;

        (ke, pe)
    }
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
    pub fn energies_mutual(&self, ipsys: &[Particle], jpsys: &[Particle]) -> (Real, Real) {
        let ((ike, ipe), (jke, jpe)) = self.compute_mutual(ipsys, jpsys);

        let ipe = ipe.iter().sum::<Real>() * 0.5;
        let ike = ike.iter().sum::<Real>() * 0.25 / self.mtot;

        let jpe = jpe.iter().sum::<Real>() * 0.5;
        let jke = jke.iter().sum::<Real>() * 0.25 / self.mtot;

        let pe = ipe + jpe;
        let ke = ike + jke;

        (ke, pe)
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

    impl Distribution<EnergySrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> EnergySrcSoA {
            EnergySrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
                rdot1: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Energy::new(1.0); // Pass mtot=1 because here we are not interested in the actual result.
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [EnergySrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [EnergyDstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [EnergySrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [EnergyDstSoA; NTILES] = [Default::default(); NTILES];
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
