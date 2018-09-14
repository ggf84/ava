#![cfg_attr(feature = "nightly-bench", feature(test))]
#![cfg(all(feature = "nightly-bench", test))]

extern crate ava;
extern crate rand;
extern crate test;

use rand::{Rng, SeedableRng, StdRng};
use std::mem::size_of;
use test::Bencher;

use ava::real::Real;
use ava::sys::particles::Particle;
use ava::sys::system::ParticleSystem;

const TILE: usize = 16 / size_of::<Real>();
const NTILES: usize = 256 / TILE;
const N: usize = NTILES * TILE;

fn init_particle_system(seed: u8, npart: usize) -> ParticleSystem {
    let mut psys = ParticleSystem::new();
    let mut rng = StdRng::from_seed([seed; 32]);
    for id in 0..npart {
        let particle = Particle {
            id: id as u64,
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
    use ava::compute::energy;

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = init_particle_system(0, N);
        b.iter(|| energy::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1, N);
        let psys2 = init_particle_system(2, N);
        b.iter(|| energy::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod acc {
    use super::*;
    use ava::compute::acc;

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = init_particle_system(0, N);
        b.iter(|| acc::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1, N);
        let psys2 = init_particle_system(2, N);
        b.iter(|| acc::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod jrk {
    use super::*;
    use ava::compute::jrk;

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = init_particle_system(0, N);
        b.iter(|| jrk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1, N);
        let psys2 = init_particle_system(2, N);
        b.iter(|| jrk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod snp {
    use super::*;
    use ava::compute::snp;

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = init_particle_system(0, N);
        b.iter(|| snp::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1, N);
        let psys2 = init_particle_system(2, N);
        b.iter(|| snp::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

#[cfg(test)]
mod crk {
    use super::*;
    use ava::compute::crk;

    #[bench]
    fn triangle(b: &mut Bencher) {
        let psys = init_particle_system(0, N);
        b.iter(|| crk::triangle(&psys.particles[..]));
    }

    #[bench]
    fn rectangle(b: &mut Bencher) {
        let psys1 = init_particle_system(1, N);
        let psys2 = init_particle_system(2, N);
        b.iter(|| crk::rectangle(&psys1.particles[..], &psys2.particles[..]));
    }
}

// -- end of file --
