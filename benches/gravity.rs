#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(all(feature = "nightly", test))]
mod bench {
    use ava::{
        gravity::{Acc0, Acc1, Acc2, Acc3, Compute, Energy},
        ics::{imf::EqualMass, sdp::Plummer, Model},
        real::Real,
        sys::ParticleSystem,
    };
    use rand::{distributions::Distribution, SeedableRng, StdRng};
    use std::mem::size_of;
    use test::Bencher;

    const TILE: usize = 16 / size_of::<Real>();
    const NTILES: usize = 256 / TILE;
    const N: usize = NTILES * TILE;

    fn init_particle_system(seed: u8, npart: usize) -> ParticleSystem {
        let mut rng = StdRng::from_seed([seed; 32]);

        let imf = EqualMass::new(1.0);
        let sdp = Plummer::new();
        let model = Model::new(npart, &imf, &sdp, 0.5, None);
        let mut psys = ParticleSystem::new();
        psys.particles = model.sample_iter(&mut rng).take(npart).collect();

        psys
    }

    mod acc0 {
        use super::*;

        #[bench]
        fn compute(b: &mut Bencher) {
            let kernel = Acc0 {};
            let psys = init_particle_system(0, N);
            b.iter(|| kernel.compute(psys.as_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let kernel = Acc0 {};
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| kernel.compute_mutual(psys1.as_slice(), psys2.as_slice()));
        }
    }

    mod acc1 {
        use super::*;

        #[bench]
        fn compute(b: &mut Bencher) {
            let kernel = Acc1 {};
            let psys = init_particle_system(0, N);
            b.iter(|| kernel.compute(psys.as_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let kernel = Acc1 {};
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| kernel.compute_mutual(psys1.as_slice(), psys2.as_slice()));
        }
    }

    mod acc2 {
        use super::*;

        #[bench]
        fn compute(b: &mut Bencher) {
            let kernel = Acc2 {};
            let psys = init_particle_system(0, N);
            b.iter(|| kernel.compute(psys.as_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let kernel = Acc2 {};
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| kernel.compute_mutual(psys1.as_slice(), psys2.as_slice()));
        }
    }

    mod acc3 {
        use super::*;

        #[bench]
        fn compute(b: &mut Bencher) {
            let kernel = Acc3 {};
            let psys = init_particle_system(0, N);
            b.iter(|| kernel.compute(psys.as_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let kernel = Acc3 {};
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| kernel.compute_mutual(psys1.as_slice(), psys2.as_slice()));
        }
    }

    mod energy {
        use super::*;

        #[bench]
        fn compute(b: &mut Bencher) {
            let kernel = Energy::new(1.0); // Pass mtot=1 because here we are not interested in the actual result.
            let psys = init_particle_system(0, N);
            b.iter(|| kernel.compute(psys.as_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let kernel = Energy::new(1.0); // Pass mtot=1 because here we are not interested in the actual result.
            let psys1 = init_particle_system(1, N);
            let psys2 = init_particle_system(2, N);
            b.iter(|| kernel.compute_mutual(psys1.as_slice(), psys2.as_slice()));
        }
    }
}

// -- end of file --
