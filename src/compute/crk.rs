use real::Real;
use sys::particles::Particle;
use super::{Aligned, Kernel, loop1, loop2, loop3, TILE};

struct Crk {}
impl Kernel for Crk {
    type SrcType = [Particle];
    type DstType = [([Real; 3], [Real; 3], [Real; 3], [Real; 3])];

    const NTILES: usize = 32;

    // flop count: 152
    fn kernel(
        &self,
        isrc: &Self::SrcType,
        jsrc: &Self::SrcType,
        idst: &mut Self::DstType,
        jdst: &mut Self::DstType,
    ) {
        let ni_tiles = (isrc.len() + TILE - 1) / TILE;
        let mut _ieps2: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _imass: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _ir0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ir1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ir2: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ir3: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia2: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ia3: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        isrc.chunks(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter().enumerate().for_each(|(i, p)| {
                _ieps2[ii][i] = p.eps2;
                _imass[ii][i] = p.mass;
                loop1(3, |k| {
                    _ir0[ii][k][i] = p.pos[k];
                    _ir1[ii][k][i] = p.vel[k];
                    _ir2[ii][k][i] = p.acc0[k];
                    _ir3[ii][k][i] = p.acc1[k];
                });
            });
        });

        let nj_tiles = (jsrc.len() + TILE - 1) / TILE;
        let mut _jeps2: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jmass: Aligned<[[Real; TILE]; Self::NTILES]> = Default::default();
        let mut _jr0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _jr1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _jr2: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _jr3: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja0: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja1: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja2: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        let mut _ja3: Aligned<[[[Real; TILE]; 3]; Self::NTILES]> = Default::default();
        jsrc.chunks(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter().enumerate().for_each(|(j, p)| {
                _jeps2[jj][j] = p.eps2;
                _jmass[jj][j] = p.mass;
                loop1(3, |k| {
                    _jr0[jj][k][j] = p.pos[k];
                    _jr1[jj][k][j] = p.vel[k];
                    _jr2[jj][k][j] = p.acc0[k];
                    _jr3[jj][k][j] = p.acc1[k];
                });
            });
        });

        const CQ21: Real = 5.0 / 3.0;
        const CQ31: Real = 8.0 / 3.0;
        const CQ32: Real = 7.0 / 3.0;
        let mut dr0: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut dr1: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut dr2: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut dr3: Aligned<[[[Real; TILE]; TILE]; 3]> = Default::default();
        let mut s0: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s1: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s2: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s3: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s11: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut s12: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut q32: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut q31: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut q21: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv2: Aligned<[[Real; TILE]; TILE]> = Default::default();
        let mut rinv3: Aligned<[[Real; TILE]; TILE]> = Default::default();

        for ii in 0..ni_tiles {
            let ieps2 = _ieps2[ii];
            let imass = _imass[ii];
            let ir0 = _ir0[ii];
            let ir1 = _ir1[ii];
            let ir2 = _ir2[ii];
            let ir3 = _ir3[ii];
            let mut ia0 = _ia0[ii];
            let mut ia1 = _ia1[ii];
            let mut ia2 = _ia2[ii];
            let mut ia3 = _ia3[ii];
            for jj in 0..nj_tiles {
                let jeps2 = &_jeps2[jj];
                let jmass = &_jmass[jj];
                let jr0 = &_jr0[jj];
                let jr1 = &_jr1[jj];
                let jr2 = &_jr2[jj];
                let jr3 = &_jr3[jj];
                let ja0 = &mut _ja0[jj];
                let ja1 = &mut _ja1[jj];
                let ja2 = &mut _ja2[jj];
                let ja3 = &mut _ja3[jj];
                loop3(3, TILE, TILE, |k, i, j| {
                    dr0[k][i][j] = ir0[k][j ^ i] - jr0[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr1[k][i][j] = ir1[k][j ^ i] - jr1[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr2[k][i][j] = ir2[k][j ^ i] - jr2[k][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr3[k][i][j] = ir3[k][j ^ i] - jr3[k][j];
                });

                loop2(TILE, TILE, |i, j| {
                    s0[i][j] = ieps2[j ^ i] + jeps2[j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s0[i][j] += dr0[k][i][j] * dr0[k][i][j];
                });
                loop2(TILE, TILE, |i, j| {
                    rinv3[i][j] = imass[j ^ i] * jmass[j];
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
                    s1[i][j] += dr0[k][i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s2[i][j] += dr0[k][i][j] * dr2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s3[i][j] += dr0[k][i][j] * dr3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s11[i][j] += dr1[k][i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    s12[i][j] += dr1[k][i][j] * dr2[k][i][j];
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
                    dr3[k][i][j] -= q32[i][j] * dr2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr3[k][i][j] -= q31[i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr2[k][i][j] -= q21[i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr3[k][i][j] -= s3[i][j] * dr0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr2[k][i][j] -= s2[i][j] * dr0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    dr1[k][i][j] -= s1[i][j] * dr0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia0[k][j ^ i] -= rinv3[i][j] * dr0[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja0[k][j] += rinv3[i][j] * dr0[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia1[k][j ^ i] -= rinv3[i][j] * dr1[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja1[k][j] += rinv3[i][j] * dr1[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia2[k][j ^ i] -= rinv3[i][j] * dr2[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja2[k][j] += rinv3[i][j] * dr2[k][i][j];
                });

                loop3(3, TILE, TILE, |k, i, j| {
                    ia3[k][j ^ i] -= rinv3[i][j] * dr3[k][i][j];
                });
                loop3(3, TILE, TILE, |k, i, j| {
                    ja3[k][j] += rinv3[i][j] * dr3[k][i][j];
                });
            }
            _ia0[ii] = ia0;
            _ia1[ii] = ia1;
            _ia2[ii] = ia2;
            _ia3[ii] = ia3;
        }

        jdst.chunks_mut(TILE).enumerate().for_each(|(jj, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(j, acc)| {
                let minv = 1.0 / _jmass[jj][j];
                loop1(3, |k| {
                    acc.0[k] += _ja0[jj][k][j] * minv;
                    acc.1[k] += _ja1[jj][k][j] * minv;
                    acc.2[k] += _ja2[jj][k][j] * minv;
                    acc.3[k] += _ja3[jj][k][j] * minv;
                });
            });
        });

        idst.chunks_mut(TILE).enumerate().for_each(|(ii, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(i, acc)| {
                let minv = 1.0 / _imass[ii][i];
                loop1(3, |k| {
                    acc.0[k] += _ia0[ii][k][i] * minv;
                    acc.1[k] += _ia1[ii][k][i] * minv;
                    acc.2[k] += _ia2[ii][k][i] * minv;
                    acc.3[k] += _ia3[ii][k][i] * minv;
                });
            });
        });
    }
}

pub fn triangle(
    src: &[Particle],
) -> (
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
    Vec<[Real; 3]>,
) {
    let mut dst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); src.len()];
    Crk {}.triangle(&src[..], &mut dst[..]);

    let mut acc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    dst.iter().for_each(|a| {
        acc.0.push(a.0);
        acc.1.push(a.1);
        acc.2.push(a.2);
        acc.3.push(a.3);
    });
    acc
}

pub fn rectangle(
    isrc: &[Particle],
    jsrc: &[Particle],
) -> (
    (
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
    ),
    (
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
    ),
) {
    let mut idst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); isrc.len()];
    let mut jdst = vec![([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]); jsrc.len()];
    Crk {}.rectangle(&isrc[..], &jsrc[..], &mut idst[..], &mut jdst[..]);

    let mut iacc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut jacc = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    idst.iter().for_each(|ia| {
        iacc.0.push(ia.0);
        iacc.1.push(ia.1);
        iacc.2.push(ia.2);
        iacc.3.push(ia.3);
    });
    jdst.iter().for_each(|ja| {
        jacc.0.push(ja.0);
        jacc.1.push(ja.1);
        jacc.2.push(ja.2);
        jacc.3.push(ja.3);
    });
    (iacc, jacc)
}

// -- end of file --
