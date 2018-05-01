#![feature(test)]

extern crate ava;
extern crate rand;
extern crate test;

use std::mem::size_of;
use rand::{Rng, SeedableRng, StdRng};

use ava::real::Real;
use ava::sys::particles::Particle;
use ava::sys::system::ParticleSystem;

const TILE: usize = 16 / size_of::<Real>();
const NTILES: usize = 64 / TILE;
const N: usize = NTILES * TILE;

fn init_particle_system(seed: usize) -> ParticleSystem {
    let mut psys = ParticleSystem::new();
    let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
    for id in 0..N {
        let particle = Particle {
            id: id,
            eps: rng.gen(),
            mass: rng.gen(),
            pos: rng.gen(),
            vel: rng.gen(),
            acc0: rng.gen(),
            acc1: rng.gen(),
            ..Default::default()
        };
        psys.particles.push(particle);
    }
    psys
}

#[cfg(test)]
mod energy {
    use super::*;
    use test::Bencher;
    use ava::compute::energy::{self, Energy, EnergyData};

    fn init_energy_data(seed: usize) -> [EnergyData; NTILES] {
        let mut data: [EnergyData; NTILES] = Default::default();
        let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
        for p in data.iter_mut() {
            p.eps = rng.gen();
            p.mass = rng.gen();
            p.r0 = rng.gen();
            p.r1 = rng.gen();
        }
        data
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Energy {};
        let mut idata = init_energy_data(0);
        let mut jdata = init_energy_data(1);
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
        let psys = init_particle_system(0);
        b.iter(|| energy::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1);
        let psys2 = init_particle_system(2);
        b.iter(|| energy::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod acc {
    use super::*;
    use test::Bencher;
    use ava::compute::acc::{self, Acc, AccData};

    fn init_acc_data(seed: usize) -> [AccData; NTILES] {
        let mut data: [AccData; NTILES] = Default::default();
        let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
        for p in data.iter_mut() {
            p.eps = rng.gen();
            p.mass = rng.gen();
            p.r0 = rng.gen();
        }
        data
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Acc {};
        let mut idata = init_acc_data(0);
        let mut jdata = init_acc_data(1);
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
        let psys = init_particle_system(0);
        b.iter(|| acc::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1);
        let psys2 = init_particle_system(2);
        b.iter(|| acc::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod jrk {
    use super::*;
    use test::Bencher;
    use ava::compute::jrk::{self, Jrk, JrkData};

    fn init_jrk_data(seed: usize) -> [JrkData; NTILES] {
        let mut data: [JrkData; NTILES] = Default::default();
        let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
        for p in data.iter_mut() {
            p.eps = rng.gen();
            p.mass = rng.gen();
            p.r0 = rng.gen();
            p.r1 = rng.gen();
        }
        data
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Jrk {};
        let mut idata = init_jrk_data(0);
        let mut jdata = init_jrk_data(1);
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
        let psys = init_particle_system(0);
        b.iter(|| jrk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1);
        let psys2 = init_particle_system(2);
        b.iter(|| jrk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod snp {
    use super::*;
    use test::Bencher;
    use ava::compute::snp::{self, Snp, SnpData};

    fn init_snp_data(seed: usize) -> [SnpData; NTILES] {
        let mut data: [SnpData; NTILES] = Default::default();
        let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
        for p in data.iter_mut() {
            p.eps = rng.gen();
            p.mass = rng.gen();
            p.r0 = rng.gen();
            p.r1 = rng.gen();
            p.r2 = rng.gen();
        }
        data
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Snp {};
        let mut idata = init_snp_data(0);
        let mut jdata = init_snp_data(1);
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
        let psys = init_particle_system(0);
        b.iter(|| snp::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1);
        let psys2 = init_particle_system(2);
        b.iter(|| snp::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod crk {
    use super::*;
    use test::Bencher;
    use ava::compute::crk::{self, Crk, CrkData};

    fn init_crk_data(seed: usize) -> [CrkData; NTILES] {
        let mut data: [CrkData; NTILES] = Default::default();
        let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);
        for p in data.iter_mut() {
            p.eps = rng.gen();
            p.mass = rng.gen();
            p.r0 = rng.gen();
            p.r1 = rng.gen();
            p.r2 = rng.gen();
            p.r3 = rng.gen();
        }
        data
    }

    #[bench]
    fn p2p(b: &mut Bencher) {
        let kernel = Crk {};
        let mut idata = init_crk_data(0);
        let mut jdata = init_crk_data(1);
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
        let psys = init_particle_system(0);
        b.iter(|| crk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1);
        let psys2 = init_particle_system(2);
        b.iter(|| crk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

// -- end of file --
