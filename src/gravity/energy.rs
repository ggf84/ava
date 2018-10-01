use super::{loop1, loop2, loop3, Compute, FromSoA, ToSoA, TILE};
use crate::{real::Real, sys::ParticleSystem};
use soa_derive::StructOfArray;

#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct Derivs<T>([T; 3], [T; 3]);

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
    ekin: [Real; TILE],
    epot: [Real; TILE],
}

#[derive(Clone, Default, Debug, PartialEq, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq"]
pub struct Src {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
    rdot1: [Real; 3],
}

#[derive(Clone, Default, Debug, PartialEq, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq"]
pub struct Dst {
    ekin: Real,
    epot: Real,
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
                    loop1(3, |k| p_src.rdot.1[k][j] = self.rdot1[jj][k]);
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
impl_kernel!(SrcSlice, DstSliceMut, SrcSoA, DstSoA, 64);

impl Kernel for Energy {
    // flop count: 28
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
                let mut s11: [[Real; TILE]; TILE] = Default::default();
                let mut rinv1: [[Real; TILE]; TILE] = Default::default();
                let mut rinv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm: [[Real; TILE]; TILE] = Default::default();
                let mut mmv2: [[Real; TILE]; TILE] = Default::default();
                let mut mm_r1: [[Real; TILE]; TILE] = Default::default();

                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.0[k][i][j] = ip_src.rdot.0[k][j ^ i] - jp_src.rdot.0[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    drdot.1[k][i][j] = ip_src.rdot.1[k][j ^ i] - jp_src.rdot.1[k][j];
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
    }
}

impl<'a, 'b: 'a> From<&'b ParticleSystem> for SrcSlice<'a> {
    fn from(ps: &'b ParticleSystem) -> Self {
        SrcSlice {
            eps: &ps.attrs.eps[..],
            mass: &ps.attrs.mass[..],
            rdot0: &ps.attrs.pos[..],
            rdot1: &ps.attrs.vel[..],
        }
    }
}

impl<S: Into<SrcSlice<'_>>> Compute<S> for Energy {
    type Output = (Vec<Real>, Vec<Real>);
    fn compute(&self, src: S) -> Self::Output {
        let src = src.into();
        let mut ke = vec![Default::default(); src.len()];
        let mut pe = vec![Default::default(); src.len()];
        let mut dst = DstSliceMut {
            ekin: &mut ke[..],
            epot: &mut pe[..],
        };

        self.triangle(&src, &mut dst);
        (ke, pe)
    }
    fn compute_mutual(&self, isrc: S, jsrc: S) -> (Self::Output, Self::Output) {
        let isrc = isrc.into();
        let jsrc = jsrc.into();
        let mut ike = vec![Default::default(); isrc.len()];
        let mut ipe = vec![Default::default(); isrc.len()];
        let mut jke = vec![Default::default(); jsrc.len()];
        let mut jpe = vec![Default::default(); jsrc.len()];
        let mut idst = DstSliceMut {
            ekin: &mut ike[..],
            epot: &mut ipe[..],
        };
        let mut jdst = DstSliceMut {
            ekin: &mut jke[..],
            epot: &mut jpe[..],
        };

        self.rectangle(&isrc, &mut idst, &jsrc, &mut jdst);
        ((ike, ipe), (jke, jpe))
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
    pub fn energies<'a, S: Into<SrcSlice<'a>>>(&self, src: S) -> (Real, Real) {
        let (ke, pe) = self.compute(src);

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
    pub fn energies_mutual<'a, S: Into<SrcSlice<'a>>>(&self, isrc: S, jsrc: S) -> (Real, Real) {
        let ((ike, ipe), (jke, jpe)) = self.compute_mutual(isrc, jsrc);

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

    impl Distribution<SrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SrcSoA {
            SrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot: Derivs(rng.gen(), rng.gen()),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Energy::new(1.0); // Pass mtot=1 because here we are not interested in the actual result.
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
