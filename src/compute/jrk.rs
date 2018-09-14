use super::{loop1, loop2, loop3, FromSoA, ToSoA, TILE};
use real::Real;
use sys::particles::Particle;

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct JrkSrcSoA {
    eps: [Real; TILE],
    mass: [Real; TILE],
    rdot0: [[Real; TILE]; 3],
    rdot1: [[Real; TILE]; 3],
}

#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone)]
struct JrkDstSoA {
    adot0: [[Real; TILE]; 3],
    adot1: [[Real; TILE]; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct JrkSrc {
    eps: Real,
    mass: Real,
    rdot0: [Real; 3],
    rdot1: [Real; 3],
}

#[derive(Debug, Default, PartialEq, StructOfArray)]
#[soa_derive = "Debug, PartialEq"]
struct JrkDst {
    adot0: [Real; 3],
    adot1: [Real; 3],
}

impl<'a> ToSoA for JrkSrcSlice<'a> {
    type SrcTypeSoA = JrkSrcSoA;
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
                    loop1(3, |k| p_src[jj].rdot1[k][j] = self.rdot1[jjj][k]);
                }
            }
        }
    }
}

impl<'a> FromSoA for JrkDstSliceMut<'a> {
    type SrcTypeSoA = JrkSrcSoA;
    type DstTypeSoA = JrkDstSoA;
    fn from_soa(&mut self, p_src: &[Self::SrcTypeSoA], p_dst: &[Self::DstTypeSoA]) {
        let n = self.len();
        let n_tiles = p_src.len();
        for jj in 0..n_tiles {
            for j in 0..TILE {
                let jjj = TILE * jj + j;
                if jjj < n {
                    let minv = 1.0 / p_src[jj].mass[j];
                    loop1(3, |k| self.adot0[jjj][k] += p_dst[jj].adot0[k][j] * minv);
                    loop1(3, |k| self.adot1[jjj][k] += p_dst[jj].adot1[k][j] * minv);
                }
            }
        }
    }
}

pub struct Jrk {}
impl_kernel!(JrkSrcSlice, JrkDstSliceMut, JrkSrcSoA, JrkDstSoA, 64);

impl Kernel for Jrk {
    // flop count: 56
    fn p2p(
        &self,
        ip_src: &JrkSrcSoA,
        ip_dst: &mut JrkDstSoA,
        jp_src: &JrkSrcSoA,
        jp_dst: &mut JrkDstSoA,
    ) {
        let mut drdot0: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut drdot1: [[[Real; TILE]; TILE]; 3] = Default::default();
        let mut s00: [[Real; TILE]; TILE] = Default::default();
        let mut s01: [[Real; TILE]; TILE] = Default::default();
        let mut q01: [[Real; TILE]; TILE] = Default::default();
        let mut rinv1: [[Real; TILE]; TILE] = Default::default();
        let mut rinv2: [[Real; TILE]; TILE] = Default::default();
        let mut mm: [[Real; TILE]; TILE] = Default::default();
        let mut mm_r3: [[Real; TILE]; TILE] = Default::default();

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
            s01[i][j] = 0.0;
        });
        loop3(3, TILE, TILE, |k, i, j| {
            s01[i][j] += drdot0[k][i][j] * drdot1[k][i][j];
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
        loop2(TILE, TILE, |i, j| {
            rinv2[i][j] *= 3.0;
        });

        loop2(TILE, TILE, |i, j| {
            q01[i][j] = rinv2[i][j] * s01[i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            drdot1[k][i][j] -= q01[i][j] * drdot0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot0[k][j ^ i] -= mm_r3[i][j] * drdot0[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot0[k][j] += mm_r3[i][j] * drdot0[k][i][j];
        });

        loop3(3, TILE, TILE, |k, i, j| {
            ip_dst.adot1[k][j ^ i] -= mm_r3[i][j] * drdot1[k][i][j];
        });
        loop3(3, TILE, TILE, |k, i, j| {
            jp_dst.adot1[k][j] += mm_r3[i][j] * drdot1[k][i][j];
        });
    }
}

pub fn triangle(psys: &[Particle]) -> (Vec<[Real; 3]>, Vec<[Real; 3]>) {
    let mut src = JrkSrcVec::with_capacity(psys.len());
    let mut dst = JrkDstVec::with_capacity(psys.len());
    for p in psys.iter() {
        src.push(JrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
        });
        dst.push(Default::default());
    }

    Jrk {}.triangle(&src.as_slice(), &mut dst.as_mut_slice());

    (dst.adot0, dst.adot1)
}

pub fn rectangle(
    ipsys: &[Particle],
    jpsys: &[Particle],
) -> (
    (Vec<[Real; 3]>, Vec<[Real; 3]>),
    (Vec<[Real; 3]>, Vec<[Real; 3]>),
) {
    let mut isrc = JrkSrcVec::with_capacity(ipsys.len());
    let mut idst = JrkDstVec::with_capacity(ipsys.len());
    for p in ipsys.iter() {
        isrc.push(JrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
        });
        idst.push(Default::default());
    }

    let mut jsrc = JrkSrcVec::with_capacity(jpsys.len());
    let mut jdst = JrkDstVec::with_capacity(jpsys.len());
    for p in jpsys.iter() {
        jsrc.push(JrkSrc {
            eps: p.eps,
            mass: p.mass,
            rdot0: p.pos,
            rdot1: p.vel,
        });
        jdst.push(Default::default());
    }

    Jrk {}.rectangle(
        &isrc.as_slice(),
        &mut idst.as_mut_slice(),
        &jsrc.as_slice(),
        &mut jdst.as_mut_slice(),
    );

    ((idst.adot0, idst.adot1), (jdst.adot0, jdst.adot1))
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

    impl Distribution<JrkSrcSoA> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> JrkSrcSoA {
            JrkSrcSoA {
                eps: rng.gen(),
                mass: rng.gen(),
                rdot0: rng.gen(),
                rdot1: rng.gen(),
            }
        }
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Jrk {};
        let mut rng = StdRng::from_seed([0; 32]);
        b.iter(|| {
            let mut ip_src: [JrkSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut ip_dst: [JrkDstSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_src: [JrkSrcSoA; NTILES] = [Default::default(); NTILES];
            let mut jp_dst: [JrkDstSoA; NTILES] = [Default::default(); NTILES];
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
