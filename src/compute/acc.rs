use real::{Aligned, Real};
use super::{Kernel, loop1, loop2, loop3};
use sys::particles::Particle;
use std::mem;

const TILE: usize = 16 / mem::size_of::<Real>();

struct Acc {}
impl Kernel<Particle, ([Real; 3],)> for Acc {
    const BLOCKSIZE: usize = 32 * TILE;

    // flop count: 27
    fn kernel(
        &self,
        isrc: &[Particle],
        jsrc: &[Particle],
        idst: &mut [([Real; 3],)],
        jdst: &mut [([Real; 3],)],
    ) {
        let ni_tiles = (isrc.len() + TILE - 1) / TILE;
        let mut _im: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ie2: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ir: [([Aligned<[Real; TILE]>; 3],); Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ia: [([Aligned<[Real; TILE]>; 3],); Self::BLOCKSIZE / TILE] = Default::default();
        isrc.chunks(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter().enumerate().for_each(|(i, p)| {
                _im[ii][i] = p.m;
                _ie2[ii][i] = p.e2;
                loop1(3, |k| {
                    _ir[ii].0[k][i] = p.r.0[k];
                });
            });
        });

        let nj_tiles = (jsrc.len() + TILE - 1) / TILE;
        let mut _jm: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _je2: [Aligned<[Real; TILE]>; Self::BLOCKSIZE / TILE] = Default::default();
        let mut _jr: [([Aligned<[Real; TILE]>; 3],); Self::BLOCKSIZE / TILE] = Default::default();
        let mut _ja: [([Aligned<[Real; TILE]>; 3],); Self::BLOCKSIZE / TILE] = Default::default();
        jsrc.chunks(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter().enumerate().for_each(|(j, p)| {
                _jm[jj][j] = p.m;
                _je2[jj][j] = p.e2;
                loop1(3, |k| {
                    _jr[jj].0[k][j] = p.r.0[k];
                });
            });
        });

        let mut dr: ([[Aligned<[Real; TILE]>; TILE]; 3],) = Default::default();
        let mut s0: [Aligned<[Real; TILE]>; TILE] = Default::default();
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
                    rinv3[i][j] *= rinv2[i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] *= rinv2[i][j].sqrt();
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia.0[k][j ^ i] -= rinv3[i][j] * dr.0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja.0[k][j] += rinv3[i][j] * dr.0[k][i][j];
                });
            }
            _ia[ii] = ia;
        }

        jdst.chunks_mut(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(j, acc)| {
                let minv = 1.0 / _jm[jj][j];
                loop1(3, |k| {
                    acc.0[k] += _ja[jj].0[k][j] * minv;
                });
            });
        });

        idst.chunks_mut(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(i, acc)| {
                let minv = 1.0 / _im[ii][i];
                loop1(3, |k| {
                    acc.0[k] += _ia[ii].0[k][i] * minv;
                });
            });
        });
    }
}

pub fn triangle(src: &[Particle]) -> Vec<([Real; 3],)> {
    let mut dst = vec![([0.0; 3],); src.len()];
    Acc {}.triangle(&src[..], &mut dst[..]);
    dst
}

pub fn rectangle(isrc: &[Particle], jsrc: &[Particle]) -> (Vec<([Real; 3],)>, Vec<([Real; 3],)>) {
    let mut idst = vec![([0.0; 3],); isrc.len()];
    let mut jdst = vec![([0.0; 3],); jsrc.len()];
    Acc {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);
    (idst, jdst)
}

// -- end of file --
