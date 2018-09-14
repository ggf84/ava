use super::{loop1, loop2, loop3, FromSoA, ToSoA, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct AccSrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct AccDstSoA {
    adot0: [[Real; TILE]; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct AccSrc {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct AccDst {
    adot0: [Real; 3],
}

impl<'a> ToSoA for AccSrcSlice<'a> {
    type SrcTypeSoA = AccSrcSoA;
    fn to_soa(&self, p_src: &mut [Self::SrcTypeSoA]) {
        let n = self.len();
        let n_tiles = p_src.len();
        for jj in 0..n_tiles {
            for j in 0..TILE {
                let jjj = TILE * jj + j;
                if jjj < n {
                    p_src[jj].eps[j] = self.eps[jjj];
                    p_src[jj].mass[j] = self.mass[jjj];
                    loop1(3, |k| p_src[jj].rdot0[k][j] = self.rdot0[jjj][k]);
                }
            }
        }
    }
}

impl<'a> FromSoA for AccDstSliceMut<'a> {
    type SrcTypeSoA = AccSrcSoA;
    type DstTypeSoA = AccDstSoA;
    fn from_soa(&mut self, p_src: &[Self::SrcTypeSoA], p_dst: &[Self::DstTypeSoA]) {
        let n = self.len();
        let n_tiles = p_src.len();
        for jj in 0..n_tiles {
            for j in 0..TILE {
                let jjj = TILE * jj + j;
                if jjj < n {
                    let minv = 1.0 / p_src[jj].mass[j];
                    loop1(3, |k| self.adot0[jjj][k] += p_dst[jj].adot0[k][j] * minv);
                }
            }
        }
    }
}

pub struct Acc {}
impl_kernel!(AccSrcSlice, AccDstSliceMut, AccSrcSoA, AccDstSoA, 64);

impl Kernel for Acc {
    // flop count: 27
    fn p2p(
        &self,
        ip_src: &AccSrcSoA,
        ip_dst: &mut AccDstSoA,
        jp_src: &AccSrcSoA,
        jp_dst: &mut AccDstSoA,
    ) {
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

        loop3(3, TILE, TILE, |k, i, j| {
            drdot0[k][i][j] = ip_src.rdot0[k][j ^ i] - jp_src.rdot0[k][j];
        });

        loop2(TILE, TILE, |i, j| {
            s00[i][j] = ip_src.eps[j ^ i] * jp_src.eps[j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s00[i][j] += drdot0[k][i][j] * drdot0[k][i][j];
        });

        loop2(TILE, TILE, |i, j| {
            mm[i][j] = ip_src.mass[j ^ i] * jp_src.mass[j];
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
            ip_dst.adot0[k][j ^ i] -= mm_r3[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot0[k][j] += mm_r3[i][j] * drdot0[k][i][j];
        });
    }
}

pub fn triangle(psys: &[Particle]) -> (Vec<[Real; 3]>,) {
    let mut src = AccSrcVec::with_capacity(psys.len());
    let mut dst = AccDstVec::with_capacity(psys.len());
    for p in psys.iter() {
        src.push(AccSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
        });
        dst.push(Default::default());
    }

    Acc {}.triangle(&src.as_slice(), &mut dst.as_mut_slice());

    (dst.adot0,)
}

pub fn rectangle(ipsys: &[Particle], jpsys: &[Particle]) -> ((Vec<[Real; 3]>,), (Vec<[Real; 3]>,)) {
    let mut isrc = AccSrcVec::with_capacity(ipsys.len());
    let mut idst = AccDstVec::with_capacity(ipsys.len());
    for p in ipsys.iter() {
        isrc.push(AccSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
        });
        idst.push(Default::default());
    }

    let mut jsrc = AccSrcVec::with_capacity(jpsys.len());
    let mut jdst = AccDstVec::with_capacity(jpsys.len());
    for p in jpsys.iter() {
        jsrc.push(AccSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
        });
        jdst.push(Default::default());
    }

    Acc {}.rectangle(
        &isrc.as_slice(),
        &mut idst.as_mut_slice(),
        &jsrc.as_slice(),
        &mut jdst.as_mut_slice(),
    );

    ((idst.adot0,), (jdst.adot0,))
}

#[cfg(all(feature = "nightly-bench", test))]
mod bench {
    extern crate test;

    use self::test::Bencher;
    use super::*;
    use rand::{
        distributions::{Distribution, Standard},
        Rng, SeedableRng, StdRng,
    };

    const NTILES: usize = 256 / TILE;

    impl Distribution<AccSrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> AccSrcSoA {
            AccSrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [AccSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [AccDstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [AccSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [AccDstSoA; NTILES] = [Default::default(); NTILES];
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
