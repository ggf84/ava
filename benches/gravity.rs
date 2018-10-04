#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(all(feature = "nightly", test))]
mod bench {
    use ava::{
        gravity::Compute,
        ics::{imf::EqualMass, sdp::Plummer, Model},
        real::Real,
        sys::ParticleSystem,
    };
    use rand::{SeedableRng, StdRng};
    use std::mem::size_of;
    use test::Bencher;

    const TILE: usize = 16 / size_of::<Real>();
    const NTILES: usize = 256 / TILE;
    const N: usize = NTILES * TILE;

    fn init_particle_system(npart: usize, seed: u8) -> ParticleSystem {
        let mut rng = StdRng::from_seed([seed; 32]);

        let imf = EqualMass::new(1.0);
        let sdp = Plummer::new();
        let model = Model::new(imf, sdp);

        ParticleSystem::from_model(npart, &model, &mut rng) // .to_standard_units(0.5, None)
    }

    mod acc0 {
        use super::*;
        use ava::gravity::acc0::{AccDot0, AccDot0Kernel};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = AccDot0::zeros(psys.len());
            let kernel = AccDot0Kernel {};

            b.iter(|| kernel.compute(&psys, &mut acc));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = AccDot0::zeros(psys1.len());
            let mut acc2 = AccDot0::zeros(psys2.len());
            let kernel = AccDot0Kernel {};

            b.iter(|| kernel.compute_mutual(&psys1, &psys2, &mut acc1, &mut acc2));
        }
    }

    mod acc1 {
        use super::*;
        use ava::gravity::acc1::{AccDot1, AccDot1Kernel};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = AccDot1::zeros(psys.len());
            let kernel = AccDot1Kernel {};

            b.iter(|| kernel.compute(&psys, &mut acc));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = AccDot1::zeros(psys1.len());
            let mut acc2 = AccDot1::zeros(psys2.len());
            let kernel = AccDot1Kernel {};

            b.iter(|| kernel.compute_mutual(&psys1, &psys2, &mut acc1, &mut acc2));
        }
    }

    mod acc2 {
        use super::*;
        use ava::gravity::acc2::{AccDot2, AccDot2Kernel};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = AccDot2::zeros(psys.len());
            let kernel = AccDot2Kernel {};

            b.iter(|| kernel.compute(&psys, &mut acc));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = AccDot2::zeros(psys1.len());
            let mut acc2 = AccDot2::zeros(psys2.len());
            let kernel = AccDot2Kernel {};

            b.iter(|| kernel.compute_mutual(&psys1, &psys2, &mut acc1, &mut acc2));
        }
    }

    mod acc3 {
        use super::*;
        use ava::gravity::acc3::{AccDot3, AccDot3Kernel};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = AccDot3::zeros(psys.len());
            let kernel = AccDot3Kernel {};

            b.iter(|| kernel.compute(&psys, &mut acc));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = AccDot3::zeros(psys1.len());
            let mut acc2 = AccDot3::zeros(psys2.len());
            let kernel = AccDot3Kernel {};

            b.iter(|| kernel.compute_mutual(&psys1, &psys2, &mut acc1, &mut acc2));
        }
    }

    mod energy {
        use super::*;
        use ava::gravity::energy::{Energy, EnergyKernel};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut energy = Energy::zeros(psys.len());
            let kernel = EnergyKernel {};

            b.iter(|| kernel.compute(&psys, &mut energy));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut energy1 = Energy::zeros(psys1.len());
            let mut energy2 = Energy::zeros(psys2.len());
            let kernel = EnergyKernel {};

            b.iter(|| kernel.compute_mutual(&psys1, &psys2, &mut energy1, &mut energy2));
        }
    }
}

// -- end of file --
