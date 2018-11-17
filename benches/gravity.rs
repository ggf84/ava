#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(all(feature = "nightly", test))]
mod bench {
    use ava::{
        gravity::Compute,
        ics::{imf::EqualMass, sdp::Plummer, Model},
        sys::ParticleSystem,
        types::{AsSlice, AsSliceMut, Real},
    };
    use rand::{rngs::SmallRng, SeedableRng};
    use std::mem::size_of;
    use test::Bencher;

    const TILE: usize = 16 / size_of::<Real>();
    const NTILES: usize = 256 / TILE;
    const N: usize = NTILES * TILE;

    fn init_particle_system(npart: usize, seed: u64) -> ParticleSystem {
        let mut rng = SmallRng::seed_from_u64(seed);

        let imf = EqualMass::new(1.0);
        let sdp = Plummer::new();
        let model = Model::new(imf, sdp);

        ParticleSystem::from_model(npart, &model, &mut rng) // .to_standard_units(0.5, None)
    }

    mod acc0 {
        use super::*;
        use ava::gravity::acc0::{AccDot0Kernel, Derivs0};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = Derivs0::zeros(psys.len());
            let kernel = AccDot0Kernel {};

            b.iter(|| kernel.compute(psys.attrs.as_slice().into(), acc.as_mut_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = Derivs0::zeros(psys1.len());
            let mut acc2 = Derivs0::zeros(psys2.len());
            let kernel = AccDot0Kernel {};

            b.iter(|| {
                kernel.compute_mutual(
                    psys1.attrs.as_slice().into(),
                    psys2.attrs.as_slice().into(),
                    acc1.as_mut_slice(),
                    acc2.as_mut_slice(),
                )
            });
        }
    }

    mod acc1 {
        use super::*;
        use ava::gravity::acc1::{AccDot1Kernel, Derivs1};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = Derivs1::zeros(psys.len());
            let kernel = AccDot1Kernel {};

            b.iter(|| kernel.compute(psys.attrs.as_slice().into(), acc.as_mut_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = Derivs1::zeros(psys1.len());
            let mut acc2 = Derivs1::zeros(psys2.len());
            let kernel = AccDot1Kernel {};

            b.iter(|| {
                kernel.compute_mutual(
                    psys1.attrs.as_slice().into(),
                    psys2.attrs.as_slice().into(),
                    acc1.as_mut_slice(),
                    acc2.as_mut_slice(),
                )
            });
        }
    }

    mod acc2 {
        use super::*;
        use ava::gravity::acc2::{AccDot2Kernel, Derivs2};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = Derivs2::zeros(psys.len());
            let kernel = AccDot2Kernel {};

            b.iter(|| kernel.compute(psys.attrs.as_slice().into(), acc.as_mut_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = Derivs2::zeros(psys1.len());
            let mut acc2 = Derivs2::zeros(psys2.len());
            let kernel = AccDot2Kernel {};

            b.iter(|| {
                kernel.compute_mutual(
                    psys1.attrs.as_slice().into(),
                    psys2.attrs.as_slice().into(),
                    acc1.as_mut_slice(),
                    acc2.as_mut_slice(),
                )
            });
        }
    }

    mod acc3 {
        use super::*;
        use ava::gravity::acc3::{AccDot3Kernel, Derivs3};

        #[bench]
        fn compute(b: &mut Bencher) {
            let psys = init_particle_system(N, 0);

            let mut acc = Derivs3::zeros(psys.len());
            let kernel = AccDot3Kernel {};

            b.iter(|| kernel.compute(psys.attrs.as_slice().into(), acc.as_mut_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut acc1 = Derivs3::zeros(psys1.len());
            let mut acc2 = Derivs3::zeros(psys2.len());
            let kernel = AccDot3Kernel {};

            b.iter(|| {
                kernel.compute_mutual(
                    psys1.attrs.as_slice().into(),
                    psys2.attrs.as_slice().into(),
                    acc1.as_mut_slice(),
                    acc2.as_mut_slice(),
                )
            });
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

            b.iter(|| kernel.compute(psys.attrs.as_slice().into(), energy.as_mut_slice()));
        }

        #[bench]
        fn compute_mutual(b: &mut Bencher) {
            let psys1 = init_particle_system(N, 1);
            let psys2 = init_particle_system(N, 2);

            let mut energy1 = Energy::zeros(psys1.len());
            let mut energy2 = Energy::zeros(psys2.len());
            let kernel = EnergyKernel {};

            b.iter(|| {
                kernel.compute_mutual(
                    psys1.attrs.as_slice().into(),
                    psys2.attrs.as_slice().into(),
                    energy1.as_mut_slice(),
                    energy2.as_mut_slice(),
                )
            });
        }
    }
}

// -- end of file --
