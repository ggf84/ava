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

trait ToSoA {
    type SoaType;
    fn to_soa(&self, ps: &mut Self::SoaType);
}

trait FromSoA {
    type SoaType;
    fn from_soa(&mut self, ps: &Self::SoaType);
}

pub trait Compute<'a> {
    type Input;
    type Output;
    fn compute(&self, src: Self::Input, dst: Self::Output);
    fn compute_mutual(
        &self,
        isrc: Self::Input,
        jsrc: Self::Input,
        idst: Self::Output,
        jdst: Self::Output,
    );
}

macro_rules! impl_kernel {
    ($scr_type:ty, $dst_type:ty, $soa_data_type:ty, $threshold:expr $(,)*) => {
        /// Kernel trait for triangle/rectangle parallel computations.
        trait Kernel: Sync {
            /// Implements mutual interaction
            fn p2p(&self, ni: usize, nj: usize, ip: &mut $soa_data_type, jp: &mut $soa_data_type);

            /// Sequential kernel
            fn kernel(
                &self,
                isrc: &$scr_type,
                jsrc: &$scr_type,
                idst: &mut $dst_type,
                jdst: &mut $dst_type,
            ) {
                let mut ip: $soa_data_type = Default::default();
                let mut jp: $soa_data_type = Default::default();

                isrc.to_soa(&mut ip);
                jsrc.to_soa(&mut jp);

                self.p2p(isrc.len(), jsrc.len(), &mut ip, &mut jp);

                jdst.from_soa(&jp);
                idst.from_soa(&ip);
            }

            /// Parallel triangle kernel
            fn triangle(&self, src: $scr_type, mut dst: $dst_type) {
                let len = src.len();
                if len > 1 {
                    let mid = len / 2;

                    let (src_lo, src_hi) = src.split_at(mid);
                    let (mut dst_lo, mut dst_hi) = dst.split_at_mut(mid);

                    rayon::join(
                        || self.triangle(src_lo.as_slice(), dst_lo.as_mut_slice()),
                        || self.triangle(src_hi.as_slice(), dst_hi.as_mut_slice()),
                    );

                    self.rectangle(
                        src_lo.as_slice(),
                        src_hi.as_slice(),
                        dst_lo.as_mut_slice(),
                        dst_hi.as_mut_slice(),
                    );
                }
            }

            /// Parallel rectangle kernel
            fn rectangle(
                &self,
                isrc: $scr_type,
                jsrc: $scr_type,
                mut idst: $dst_type,
                mut jdst: $dst_type,
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
                        || {
                            self.rectangle(
                                isrc_lo.as_slice(),
                                jsrc_hi.as_slice(),
                                idst_lo.as_mut_slice(),
                                jdst_hi.as_mut_slice(),
                            )
                        },
                        || {
                            self.rectangle(
                                isrc_hi.as_slice(),
                                jsrc_lo.as_slice(),
                                idst_hi.as_mut_slice(),
                                jdst_lo.as_mut_slice(),
                            )
                        },
                    );
                    rayon::join(
                        || {
                            self.rectangle(
                                isrc_lo.as_slice(),
                                jsrc_lo.as_slice(),
                                idst_lo.as_mut_slice(),
                                jdst_lo.as_mut_slice(),
                            )
                        },
                        || {
                            self.rectangle(
                                isrc_hi.as_slice(),
                                jsrc_hi.as_slice(),
                                idst_hi.as_mut_slice(),
                                jdst_hi.as_mut_slice(),
                            )
                        },
                    );
                } else if ilen > 0 && jlen > 0 {
                    self.kernel(&isrc, &jsrc, &mut idst, &mut jdst);
                }
            }
        }
    };
}

pub mod acc0;
pub mod acc1;
pub mod acc2;
pub mod acc3;
pub mod energy;

// -- end of file --
