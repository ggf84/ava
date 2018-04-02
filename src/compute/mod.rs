pub mod acc;
pub mod phi;

use rayon;

/// Kernel methods for triangle/rectangle parallel computations.
trait Kernel<SrcType, DstType>
where
    Self: Sync,
    SrcType: Sync,
    DstType: Send,
{
    /// Fallback to sequential for arrays less than BLOCKSIZE in length
    const BLOCKSIZE: usize;

    /// Sequential kernel
    fn kernel(
        &self,
        isrc: &[SrcType],
        jsrc: &[SrcType],
        idst: &mut [DstType],
        jdst: &mut [DstType],
    );

    /// Parallel triangle kernel
    fn triangle(&self, src: &[SrcType], dst: &mut [DstType]) {
        let len = dst.len();
        if len > 1 {
            let mid = len / 2;
            let (src_lo, src_hi) = src.split_at(mid);
            let (dst_lo, dst_hi) = dst.split_at_mut(mid);

            self.rectangle(src_lo, src_hi, dst_lo, dst_hi);

            rayon::join(
                || self.triangle(src_lo, dst_lo),
                || self.triangle(src_hi, dst_hi),
            );
        }
    }

    /// Parallel rectangle kernel
    fn rectangle(
        &self,
        isrc: &[SrcType],
        jsrc: &[SrcType],
        idst: &mut [DstType],
        jdst: &mut [DstType],
    ) {
        let ilen = idst.len();
        let jlen = jdst.len();
        if ilen > Self::BLOCKSIZE || jlen > Self::BLOCKSIZE {
            let imid = ilen / 2;
            let jmid = jlen / 2;

            let (isrc_lo, isrc_hi) = isrc.split_at(imid);
            let (jsrc_lo, jsrc_hi) = jsrc.split_at(jmid);

            let (idst_lo, idst_hi) = idst.split_at_mut(imid);
            let (jdst_lo, jdst_hi) = jdst.split_at_mut(jmid);

            rayon::join(
                || self.rectangle(isrc_lo, jsrc_lo, idst_lo, jdst_lo),
                || self.rectangle(isrc_hi, jsrc_hi, idst_hi, jdst_hi),
            );
            rayon::join(
                || self.rectangle(isrc_lo, jsrc_hi, idst_lo, jdst_hi),
                || self.rectangle(isrc_hi, jsrc_lo, idst_hi, jdst_lo),
            );
        } else {
            self.kernel(isrc, jsrc, idst, jdst);
        }
    }
}

fn loop1<F>(n: usize, mut f: F)
where
    F: FnMut(usize),
{
    for k in 0..n {
        f(k);
    }
}

fn loop2<F>(ni: usize, nj: usize, mut f: F)
where
    F: FnMut(usize, usize),
{
    for i in 0..ni {
        for j in 0..nj {
            f(i, j);
        }
    }
}

fn loop3<F>(ni: usize, nj: usize, nk: usize, mut f: F)
where
    F: FnMut(usize, usize, usize),
{
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                f(i, j, k);
            }
        }
    }
}

// -- end of file --
