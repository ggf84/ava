#![feature(test)]

extern crate ava;
extern crate rand;
extern crate test;

use std::mem::size_of;
use rand::{Rng, SeedableRng, StdRng};

use ava::real::Real;
use ava::sys::particles::Particle;
use ava::sys::system::ParticleSystem;

fn create_particle_system(n: usize, seed: usize) -> ParticleSystem {
    let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);

    let mut psys = ParticleSystem::new();
    for id in 0..n {
        let eps = rng.gen_range::<Real>(0.0, 1.0);
        let mass = rng.gen_range::<Real>(0.0, 1.0);
        let pos = [
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
        ];
        let vel = [
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
        ];
        let acc0 = [
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
        ];
        let acc1 = [
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
            rng.gen_range::<Real>(-1.0, 1.0),
        ];
        let particle = Particle {
            id: id,
            eps: eps,
            mass: mass,
            pos: pos,
            vel: vel,
            acc0: acc0,
            acc1: acc1,
            ..Default::default()
        };
        psys.particles.push(particle);
    }
    psys
}

const TILE: usize = 16 / size_of::<Real>();
const NTILES: usize = 64 / TILE;
const N: usize = NTILES * TILE;

#[cfg(test)]
mod energy {
    use super::*;
    use test::Bencher;
    use ava::compute::energy::{self, Energy, EnergyData};

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Energy {};
        let mut idata: [EnergyData; NTILES] = Default::default();
        let mut jdata: [EnergyData; NTILES] = Default::default();
        b.iter(|| {
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&mut idata[ii], &mut jdata[jj]);
                }
            }
        });
        println!("{:?}", (idata, jdata));
    }

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = create_particle_system(N, 0);
        b.iter(|| energy::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = create_particle_system(N, 1);
        let psys2 = create_particle_system(N, 2);
        b.iter(|| energy::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod acc {
    use super::*;
    use test::Bencher;
    use ava::compute::acc::{self, Acc, AccData};

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc {};
        let mut idata: [AccData; NTILES] = Default::default();
        let mut jdata: [AccData; NTILES] = Default::default();
        b.iter(|| {
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&mut idata[ii], &mut jdata[jj]);
                }
            }
        });
        println!("{:?}", (idata, jdata));
    }

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = create_particle_system(N, 0);
        b.iter(|| acc::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = create_particle_system(N, 1);
        let psys2 = create_particle_system(N, 2);
        b.iter(|| acc::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod jrk {
    use super::*;
    use test::Bencher;
    use ava::compute::jrk::{self, Jrk, JrkData};

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Jrk {};
        let mut idata: [JrkData; NTILES] = Default::default();
        let mut jdata: [JrkData; NTILES] = Default::default();
        b.iter(|| {
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&mut idata[ii], &mut jdata[jj]);
                }
            }
        });
        println!("{:?}", (idata, jdata));
    }

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = create_particle_system(N, 0);
        b.iter(|| jrk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = create_particle_system(N, 1);
        let psys2 = create_particle_system(N, 2);
        b.iter(|| jrk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod snp {
    use super::*;
    use test::Bencher;
    use ava::compute::snp::{self, Snp, SnpData};

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Snp {};
        let mut idata: [SnpData; NTILES] = Default::default();
        let mut jdata: [SnpData; NTILES] = Default::default();
        b.iter(|| {
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&mut idata[ii], &mut jdata[jj]);
                }
            }
        });
        println!("{:?}", (idata, jdata));
    }

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = create_particle_system(N, 0);
        b.iter(|| snp::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = create_particle_system(N, 1);
        let psys2 = create_particle_system(N, 2);
        b.iter(|| snp::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod crk {
    use super::*;
    use test::Bencher;
    use ava::compute::crk::{self, Crk, CrkData};

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Crk {};
        let mut idata: [CrkData; NTILES] = Default::default();
        let mut jdata: [CrkData; NTILES] = Default::default();
        b.iter(|| {
            for ii in 0..NTILES {
                for jj in 0..NTILES {
                    kernel.p2p(&mut idata[ii], &mut jdata[jj]);
                }
            }
        });
        println!("{:?}", (idata, jdata));
    }

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = create_particle_system(N, 0);
        b.iter(|| crk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = create_particle_system(N, 1);
        let psys2 = create_particle_system(N, 2);
        b.iter(|| crk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

// -- end of file --
