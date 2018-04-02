use real::{Aligned, Real};
use super::{Kernel, loop1, loop2, loop3};
use sys::particles::Particle;
use std::mem;

const TILE: usize = 16 / mem::size_of::<Real>();

struct Crk {}
impl Kernel<Particle, ([Real; 3], [Real; 3], [Real; 3], [Real; 3])> for Crk {
    const BLOCKSIZE: usize = 32 * TILE;

    // flop count: 152
    fn kernel(
        &self,
        isrc: &[Particle],
        jsrc: &[Particle],
        idst: &mut [([Real; 3], [Real; 3], [Real; 3], [Real; 3])],
        jdst: &mut [([Real; 3], [Real; 3], [Real; 3], [Real; 3])],
    ) {
        let ni_tiles = (isrc.len() + TILE - 1) / TILE;
        let mut _im: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ie2: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ir: [(
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
        ); Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ia: [(
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
        ); Self::BLOCKSIZE / TILE] = Default::default();
        isrc.chunks(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter().enumerate().for_each(|(i, p)| {
                _im[ii][i] = p.m;
                _ie2[ii][i] = p.e2;
                loop1(3, |k| {
                    _ir[ii].0[k][i] = p.r.0[k];
                    _ir[ii].1[k][i] = p.r.1[k];
                    _ir[ii].2[k][i] = p.r.2[k];
                    _ir[ii].3[k][i] = p.r.3[k];
                });
            });
        });

        let nj_tiles = (jsrc.len() + TILE - 1) / TILE;
        let mut _jm: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _je2: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _jr: [(
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
        ); Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ja: [(
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
            [Aligned<[Real; TILE]>; 3],
        ); Self::BLOCKSIZE / TILE] = Default::default();
        jsrc.chunks(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter().enumerate().for_each(|(j, p)| {
                _jm[jj][j] = p.m;
                _je2[jj][j] = p.e2;
                loop1(3, |k| {
                    _jr[jj].0[k][j] = p.r.0[k];
                    _jr[jj].1[k][j] = p.r.1[k];
                    _jr[jj].2[k][j] = p.r.2[k];
                    _jr[jj].3[k][j] = p.r.3[k];
                });
            });
        });

        const CQ21: Real = 5.0 / 3.0;
        const CQ31: Real = 8.0 / 3.0;
        const CQ32: Real = 7.0 / 3.0;
        let mut dr: (
            [[Aligned<[Real; TILE]>; TILE]; 3],
            [[Aligned<[Real; TILE]>; TILE]; 3],
            [[Aligned<[Real; TILE]>; TILE]; 3],
            [[Aligned<[Real; TILE]>; TILE]; 3],
        ) = Default::default();
        let mut s0: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut s1: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut s2: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut s3: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut s11: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut s12: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut q32: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut q31: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut q21: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut rinv2: [Aligned<[Real; TILE]>; TILE] = Default::default();
        let mut rinv3: [Aligned<[Real; TILE]>; TILE] = Default::default();

        for ii in 0..ni_tiles {
            let im = _im[ii];
            let ie2 = _ie2[ii];
            let ir = _ir[ii];
            let mut ia = _ia[ii];
            for jj in 0..nj_tiles {
                let jm = &_jm[jj];
                let je2 = &_je2[jj];
                let jr = &_jr[jj];
                let ja = &mut _ja[jj];
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.0[k][i][j] = ir.0[k][j ^ i] - jr.0[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.1[k][i][j] = ir.1[k][j ^ i] - jr.1[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.2[k][i][j] = ir.2[k][j ^ i] - jr.2[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.3[k][i][j] = ir.3[k][j ^ i] - jr.3[k][j];
                });

                loop2(TILE, TILE, |i, j| {
                    s0[i][j] = ie2[j ^ i] + je2[j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s0[i][j] += dr.0[k][i][j] * dr.0[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] = im[j ^ i] * jm[j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] = s0[i][j].recip();
                });

                loop2(TILE, TILE, |i, j| {
                    s1[i][j] = 0.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s2[i][j] = 0.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s3[i][j] = 0.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s11[i][j] = 0.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s12[i][j] = 0.0;
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s1[i][j] += dr.0[k][i][j] * dr.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s2[i][j] += dr.0[k][i][j] * dr.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s3[i][j] += dr.0[k][i][j] * dr.3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s11[i][j] += dr.1[k][i][j] * dr.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s12[i][j] += dr.1[k][i][j] * dr.2[k][i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] *= rinv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] *= rinv2[i][j].sqrt();
                });

                loop2(TILE, TILE, |i, j| {
                    s2[i][j] += s11[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s3[i][j] += 3.0 * s12[i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    rinv2[i][j] *= 3.0;
                });
                loop2(TILE, TILE, |i, j| {
                    s1[i][j] *= rinv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s2[i][j] *= rinv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s3[i][j] *= rinv2[i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    s3[i][j] -= CQ31 * s1[i][j] * s2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s2[i][j] -= CQ21 * s1[i][j] * s1[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    s3[i][j] -= CQ32 * s1[i][j] * s2[i][j];
                });

                loop2(TILE, TILE, |i, j| {
                    q32[i][j] = 3.0 * s1[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    q31[i][j] = 3.0 * s2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    q21[i][j] = 2.0 * s1[i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    dr.3[k][i][j] -= q32[i][j] * dr.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.3[k][i][j] -= q31[i][j] * dr.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.2[k][i][j] -= q21[i][j] * dr.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.3[k][i][j] -= s3[i][j] * dr.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.2[k][i][j] -= s2[i][j] * dr.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr.1[k][i][j] -= s1[i][j] * dr.0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia.0[k][j ^ i] -= rinv3[i][j] * dr.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja.0[k][j] += rinv3[i][j] * dr.0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia.1[k][j ^ i] -= rinv3[i][j] * dr.1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja.1[k][j] += rinv3[i][j] * dr.1[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia.2[k][j ^ i] -= rinv3[i][j] * dr.2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja.2[k][j] += rinv3[i][j] * dr.2[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia.3[k][j ^ i] -= rinv3[i][j] * dr.3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja.3[k][j] += rinv3[i][j] * dr.3[k][i][j];
                });
            }
            _ia[ii] = ia;
        }

        jdst.chunks_mut(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(j, acc)| {
                let minv = 1.0 / _jm[jj][j];
                loop1(3, |k| {
                    acc.0[k] += _ja[jj].0[k][j] * minv;
                    acc.1[k] += _ja[jj].1[k][j] * minv;
                    acc.2[k] += _ja[jj].2[k][j] * minv;
                    acc.3[k] += _ja[jj].3[k][j] * minv;
                });
            });
        });

        idst.chunks_mut(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(i, acc)| {
                let minv = 1.0 / _im[ii][i];
                loop1(3, |k| {
                    acc.0[k] += _ia[ii].0[k][i] * minv;
                    acc.1[k] += _ia[ii].1[k][i] * minv;
                    acc.2[k] += _ia[ii].2[k][i] * minv;
                    acc.3[k] += _ia[ii].3[k][i] * minv;
                });
            });
        });
    }
}

pub fn triangle(src: &[Particle]) -> Vec<([Real; 3], [Real; 3], [Real; 3], [Real; 3])> {
    let mut dst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); src.len()];
    Crk {}.triangle(&src[..], &mut dst[..]);
    dst
}

pub fn rectangle(
    isrc: &[Particle],
    jsrc: &[Particle],
) -> (
    Vec<([Real; 3], [Real; 3], [Real; 3], [Real; 3])>,
    Vec<([Real; 3], [Real; 3], [Real; 3], [Real; 3])>,
) {
    let mut idst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); isrc.len()];
    let mut jdst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); jsrc.len()];
    Crk {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);
    (idst, jdst)
}

// -- end of file --
