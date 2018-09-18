use crate::real::Real;
use std::mem::size_of;

const TILE: usize = 16 / size_of::<Real>();

trait ToSoA {
    type SrcTypeSoA;
    fn to_soa(&self, p_src: &mut [Self::SrcTypeSoA]);
}

trait FromSoA {
    type SrcTypeSoA;
    type DstTypeSoA;
    fn from_soa(&mut self, p_src: &[Self::SrcTypeSoA], p_dst: &[Self::DstTypeSoA]);
}

macro_rules! impl_kernel {
    (
        $scr_type:ident, $dst_type:ident, $scr_type_soa:ident, $dst_type_soa:ident, $threshold:expr $(,)*
    ) => {
        /// Kernel trait for triangle/rectangle parallel computations.
        trait Kernel: Sync {
            /// Implements mutual interaction
            fn p2p(
                &self,
                ip_src: &$scr_type_soa,
                ip_dst: &mut $dst_type_soa,
                jp_src: &$scr_type_soa,
                jp_dst: &mut $dst_type_soa,
            );

            /// Sequential kernel
            fn kernel(
                &self,
                isrc: &$scr_type<'_>,
                idst: &mut $dst_type<'_>,
                jsrc: &$scr_type<'_>,
                jdst: &mut $dst_type<'_>,
            ) {
                const NTILES: usize = $threshold / TILE;
                let mut ip_src: [$scr_type_soa; NTILES] = [Default::default(); NTILES];
                let mut ip_dst: [$dst_type_soa; NTILES] = [Default::default(); NTILES];
                let mut jp_src: [$scr_type_soa; NTILES] = [Default::default(); NTILES];
                let mut jp_dst: [$dst_type_soa; NTILES] = [Default::default(); NTILES];

                isrc.to_soa(&mut ip_src[..]);
                jsrc.to_soa(&mut jp_src[..]);

                let ni_tiles = (isrc.len() + TILE - 1) / TILE;
                let nj_tiles = (jsrc.len() + TILE - 1) / TILE;
                for ii in 0..ni_tiles {
                    for jj in 0..nj_tiles {
                        self.p2p(&ip_src[ii], &mut ip_dst[ii], &jp_src[jj], &mut jp_dst[jj]);
                    }
                }

                jdst.from_soa(&jp_src[..], &jp_dst[..]);
                idst.from_soa(&ip_src[..], &ip_dst[..]);
            }

            /// Parallel triangle kernel
            fn triangle(&self, src: &$scr_type<'_>, dst: &mut $dst_type<'_>) {
                let len = src.len();
                if len > 1 {
                    let mid = len / 2;

                    let (src_lo, src_hi) = src.split_at(mid);
                    let (mut dst_lo, mut dst_hi) = dst.split_at_mut(mid);

                    rayon::join(
                        || self.triangle(&src_lo, &mut dst_lo),
                        || self.triangle(&src_hi, &mut dst_hi),
                    );

                    self.rectangle(&src_lo, &mut dst_lo, &src_hi, &mut dst_hi);
                }
            }

            /// Parallel rectangle kernel
            fn rectangle(
                &self,
                isrc: &$scr_type<'_>,
                idst: &mut $dst_type<'_>,
                jsrc: &$scr_type<'_>,
                jdst: &mut $dst_type<'_>,
            ) {
                let ilen = isrc.len();
                let jlen = jsrc.len();
                if ilen > $threshold || jlen > $threshold {
                    let imid = ilen / 2;
                    let jmid = jlen / 2;

                    let (isrc_lo, isrc_hi) = isrc.split_at(imid);
                    let (mut idst_lo, mut idst_hi) = idst.split_at_mut(imid);

                    let (jsrc_lo, jsrc_hi) = jsrc.split_at(jmid);
                    let (mut jdst_lo, mut jdst_hi) = jdst.split_at_mut(jmid);

                    rayon::join(
                        || self.rectangle(&isrc_lo, &mut idst_lo, &jsrc_hi, &mut jdst_hi),
                        || self.rectangle(&isrc_hi, &mut idst_hi, &jsrc_lo, &mut jdst_lo),
                    );
                    rayon::join(
                        || self.rectangle(&isrc_lo, &mut idst_lo, &jsrc_lo, &mut jdst_lo),
                        || self.rectangle(&isrc_hi, &mut idst_hi, &jsrc_hi, &mut jdst_hi),
                    );
                } else if ilen > 0 && jlen > 0 {
                    self.kernel(isrc, idst, jsrc, jdst);
                }
            }
        }
    };
}

#[inline]
fn loop1<F>(nk: usize, mut f: F)
where
    F: FnMut(usize),
{
    for k in 0..nk {
        f(k);
    }
}

#[inline]
fn loop2<F>(ni: usize, nj: usize, mut f: F)
where
    F: FnMut(usize, usize),
{
    loop1(ni, |i| loop1(nj, |j| f(i, j)));
}

#[inline]
fn loop3<F>(ni: usize, nj: usize, nk: usize, mut f: F)
where
    F: FnMut(usize, usize, usize),
{
    loop1(ni, |i| loop1(nj, |j| loop1(nk, |k| f(i, j, k))));
}

mod acc0;
mod acc1;
mod acc2;
mod acc3;
mod energy;

pub use self::acc0::Acc0;
pub use self::acc1::Acc1;
pub use self::acc2::Acc2;
pub use self::acc3::Acc3;
pub use self::energy::Energy;

// -- end of file --
