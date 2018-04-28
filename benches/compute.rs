#![feature(test)]

extern crate ava;
extern crate rand;
extern crate test;

use test::Bencher;
use rand::{Rng, SeedableRng, StdRng};

use ava::real::Real;
use ava::sys::particles::Particle;
use ava::sys::system::ParticleSystem;
use ava::compute;

fn create_particle_system(n: usize, seed: usize) -> ParticleSystem {
    let mut rng: StdRng = SeedableRng::from_seed(&[seed][..]);

    let mut ps = ParticleSystem::new();
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
        ps.particles.push(particle);
    }

    ps
}

const N: usize = 1024;

#[bench]
fn energy_triangle(b: &mut Bencher) {
    let ps = create_particle_system(N, 0);
    b.iter(|| compute::energy::triangle(&ps.particles[..]));
}

#[bench]
fn energy_rectangle(b: &mut Bencher) {
    let ps1 = create_particle_system(N, 1);
    let ps2 = create_particle_system(N / 2, 2);
    b.iter(|| compute::energy::rectangle(&ps1.particles[..], &ps2.particles[..]));
}

#[bench]
fn acc_triangle(b: &mut Bencher) {
    let ps = create_particle_system(N, 0);
    b.iter(|| compute::acc::triangle(&ps.particles[..]));
}

#[bench]
fn acc_rectangle(b: &mut Bencher) {
    let ps1 = create_particle_system(N, 1);
    let ps2 = create_particle_system(N / 2, 2);
    b.iter(|| compute::acc::rectangle(&ps1.particles[..], &ps2.particles[..]));
}

#[bench]
fn jrk_triangle(b: &mut Bencher) {
    let ps = create_particle_system(N, 0);
    b.iter(|| compute::jrk::triangle(&ps.particles[..]));
}

#[bench]
fn jrk_rectangle(b: &mut Bencher) {
    let ps1 = create_particle_system(N, 1);
    let ps2 = create_particle_system(N / 2, 2);
    b.iter(|| compute::jrk::rectangle(&ps1.particles[..], &ps2.particles[..]));
}

#[bench]
fn snp_triangle(b: &mut Bencher) {
    let ps = create_particle_system(N, 0);
    b.iter(|| compute::snp::triangle(&ps.particles[..]));
}

#[bench]
fn snp_rectangle(b: &mut Bencher) {
    let ps1 = create_particle_system(N, 1);
    let ps2 = create_particle_system(N / 2, 2);
    b.iter(|| compute::snp::rectangle(&ps1.particles[..], &ps2.particles[..]));
}

#[bench]
fn crk_triangle(b: &mut Bencher) {
    let ps = create_particle_system(N, 0);
    b.iter(|| compute::crk::triangle(&ps.particles[..]));
}

#[bench]
fn crk_rectangle(b: &mut Bencher) {
    let ps1 = create_particle_system(N, 1);
    let ps2 = create_particle_system(N / 2, 2);
    b.iter(|| compute::crk::rectangle(&ps1.particles[..], &ps2.particles[..]));
}

// -- end of file --
