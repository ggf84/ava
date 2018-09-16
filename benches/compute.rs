#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(all(feature = "nightly", test))]
mod bench {
    use ava::{
        compute,
        real::Real,
        sys::{particles::Particle, system::ParticleSystem},
    };
    use rand::{Rng, SeedableRng, StdRng};
    use std::mem::size_of;
    use test::Bencher;

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

    mod energy {
        use super::*;

        #[bench]
        fn triangle(b: &mut Bencher) {
            let psys = init_particle_system(0, N);
            b.iter(|| compute::energy::triangle(&psys.particles[..]));
        }

        #[bench]
        fn rectangle(b: &mut Bencher) {
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| compute::energy::rectangle(&psys1.particles[..], &psys2.particles[..]));
        }
    }

    mod acc {
        use super::*;

        #[bench]
        fn triangle(b: &mut Bencher) {
            let psys = init_particle_system(0, N);
            b.iter(|| compute::acc::triangle(&psys.particles[..]));
        }

        #[bench]
        fn rectangle(b: &mut Bencher) {
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| compute::acc::rectangle(&psys1.particles[..], &psys2.particles[..]));
        }
    }

    mod jrk {
        use super::*;

        #[bench]
        fn triangle(b: &mut Bencher) {
            let psys = init_particle_system(0, N);
            b.iter(|| compute::jrk::triangle(&psys.particles[..]));
        }

        #[bench]
        fn rectangle(b: &mut Bencher) {
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| compute::jrk::rectangle(&psys1.particles[..], &psys2.particles[..]));
        }
    }

    mod snp {
        use super::*;

        #[bench]
        fn triangle(b: &mut Bencher) {
            let psys = init_particle_system(0, N);
            b.iter(|| compute::snp::triangle(&psys.particles[..]));
        }

        #[bench]
        fn rectangle(b: &mut Bencher) {
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| compute::snp::rectangle(&psys1.particles[..], &psys2.particles[..]));
        }
    }

    mod crk {
        use super::*;

        #[bench]
        fn triangle(b: &mut Bencher) {
            let psys = init_particle_system(0, N);
            b.iter(|| compute::crk::triangle(&psys.particles[..]));
        }

        #[bench]
        fn rectangle(b: &mut Bencher) {
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| compute::crk::rectangle(&psys1.particles[..], &psys2.particles[..]));
        }
    }
}

// -- end of file --
