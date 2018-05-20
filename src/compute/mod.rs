use rayon;
use real::Real;
use std::mem::size_of;

pub mod acc;
pub mod crk;
pub mod energy;
pub mod jrk;
pub mod snp;

const TILE: usize = 16 / size_of::<Real>();

trait SplitAt {
    type Output: ?Sized;
    fn len(&self) -> usize;
    fn split_at(&self, mid: usize) -> (&Self::Output, &Self::Output);
}

trait SplitAtMut {
    type Output: ?Sized;
    fn len(&self) -> usize;
    fn split_at_mut(&mut self, mid: usize) -> (&mut Self::Output, &mut Self::Output);
}

impl<T> SplitAt for [T] {
    type Output = [T];
    fn len(&self) -> usize {
        self.len()
    }
    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.split_at(mid)
    }
}

impl<T> SplitAtMut for [T] {
    type Output = [T];
    fn len(&self) -> usize {
        self.len()
    }
    fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        self.split_at_mut(mid)
    }
}

/// Kernel trait for triangle/rectangle parallel computations.
trait Kernel: Sync {
    type SrcType: ?Sized + Sync + SplitAt<Output = Self::SrcType>;
    type DstType: ?Sized + Send + SplitAtMut<Output = Self::DstType>;

    /// Fallback to sequential for arrays less than or equal to TILE * NTILES in length
    const NTILES: usize;

    /// Sequential kernel
    fn kernel(
        &self,
        isrc: &Self::SrcType,
        jsrc: &Self::SrcType,
        idst: &mut Self::DstType,
        jdst: &mut Self::DstType,
    );

    /// Parallel triangle kernel
    fn triangle(&self, src: &Self::SrcType, dst: &mut Self::DstType) {
        let len = src.len();
        if len > 1 {
            let mid = len / 2;
            let (src_lo, src_hi) = src.split_at(mid);
            let (dst_lo, dst_hi) = dst.split_at_mut(mid);

            rayon::join(
                || self.triangle(src_lo, dst_lo),
                || self.triangle(src_hi, dst_hi),
            );

            self.rectangle(src_lo, src_hi, dst_lo, dst_hi);
        }
    }

    /// Parallel rectangle kernel
    fn rectangle(
        &self,
        isrc: &Self::SrcType,
        jsrc: &Self::SrcType,
        idst: &mut Self::DstType,
        jdst: &mut Self::DstType,
    ) {
        let ilen = isrc.len();
        let jlen = jsrc.len();
        if ilen > TILE * Self::NTILES || jlen > TILE * Self::NTILES {
            let imid = ilen / 2;
            let jmid = jlen / 2;

            let (isrc_lo, isrc_hi) = isrc.split_at(imid);
            let (jsrc_lo, jsrc_hi) = jsrc.split_at(jmid);

            let (idst_lo, idst_hi) = idst.split_at_mut(imid);
            let (jdst_lo, jdst_hi) = jdst.split_at_mut(jmid);

            rayon::join(
                || self.rectangle(isrc_lo, jsrc_hi, idst_lo, jdst_hi),
                || self.rectangle(isrc_hi, jsrc_lo, idst_hi, jdst_lo),
            );
            rayon::join(
                || self.rectangle(isrc_lo, jsrc_lo, idst_lo, jdst_lo),
                || self.rectangle(isrc_hi, jsrc_hi, idst_hi, jdst_hi),
            );
        } else if ilen > 0 && jlen > 0 {
            self.kernel(isrc, jsrc, idst, jdst);
        }
    }
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

// -- end of file --
