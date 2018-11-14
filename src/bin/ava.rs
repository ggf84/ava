use ava::{
    gravity::{
        acc0::{AccDot0Kernel, Derivs0},
        acc1::{AccDot1Kernel, Derivs1},
        acc2::{AccDot2Kernel, Derivs2},
        acc3::{AccDot3Kernel, Derivs3},
        energy::{Energy, EnergyKernel},
        Compute,
    },
    ics::{
        imf::{EqualMass, Maschberger2013},
        sdp::{Dehnen0, Dehnen1, Dehnen12, Dehnen2, Dehnen32, Plummer},
        Model,
    },
    sys::ParticleSystem,
    types::{AsSlice, AsSliceMut, Len},
};
use itertools::izip;
use rand::{SeedableRng, StdRng};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    time::Instant,
};

fn main() -> Result<(), std::io::Error> {
    let seed = [0; 32];
    let mut rng = StdRng::from_seed(seed);

    let npart = 256;
    let imf = EqualMass::new(1.0);
    // let imf = Maschberger2013::new(0.01, 150.0);
    // let sdp = Plummer::new();
    let sdp = Dehnen1::new();
    let model = Model::new(imf, sdp);

    let mut psys =
        ParticleSystem::from_model(4 * npart, &model, &mut rng).into_standard_units(0.5, None);

    let file = File::create("cluster.txt")?;
    let mut writer = BufWriter::new(file);
    for (&id, &eps, &mass, &pos, &vel) in izip!(
        &psys.attrs.id,
        &psys.attrs.eps,
        &psys.attrs.mass,
        &psys.attrs.pos,
        &psys.attrs.vel
    ) {
        writeln!(
            &mut writer,
            "{} {} {} \
             {} {} {} \
             {} {} {}",
            id, eps, mass, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
        )?;
    }

    let psys1 =
        ParticleSystem::from_model(7 * npart, &model, &mut rng).into_standard_units(0.5, None);
    let psys2 =
        ParticleSystem::from_model(3 * npart, &model, &mut rng).into_standard_units(0.5, None);

    let mut iacc_12 = Derivs0::zeros(psys1.len());
    let mut jacc_12 = Derivs0::zeros(psys2.len());
    let mut iacc_21 = Derivs0::zeros(psys2.len());
    let mut jacc_21 = Derivs0::zeros(psys1.len());
    AccDot0Kernel {}.compute_mutual(
        psys1.attrs.as_slice().into(),
        psys2.attrs.as_slice().into(),
        iacc_12.as_mut_slice(),
        jacc_12.as_mut_slice(),
    );
    AccDot0Kernel {}.compute_mutual(
        psys2.attrs.as_slice().into(),
        psys1.attrs.as_slice().into(),
        iacc_21.as_mut_slice(),
        jacc_21.as_mut_slice(),
    );
    assert!(iacc_21 == jacc_12);
    assert!(iacc_12 == jacc_21);
    let mut ftot0 = [0.0; 3];
    for (i, m) in psys1.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * iacc_12.dot0[i][k];
        }
    }
    for (j, m) in psys2.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * jacc_12.dot0[j][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("iacc_12: {:?}", &iacc_12.dot0[..2]);
    eprintln!("jacc_21: {:?}", &jacc_21.dot0[..2]);
    eprintln!("jacc_12: {:?}", &jacc_12.dot0[..2]);
    eprintln!("iacc_21: {:?}", &iacc_21.dot0[..2]);
    eprintln!("");

    let mut acc = Derivs0::zeros(psys.len());
    AccDot0Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
    let mut ftot0 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.dot0[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("acc.dot0: {:?}", &acc.dot0[..2]);
    eprintln!("");

    let mut acc = Derivs1::zeros(psys.len());
    AccDot1Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.dot0[i][k];
            ftot1[k] += m * acc.dot1[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("acc.dot0: {:?}", &acc.dot0[..2]);
    eprintln!("acc.dot1: {:?}", &acc.dot1[..2]);
    eprintln!("");

    psys.attrs.acc0 = acc.dot0;
    let mut acc = Derivs2::zeros(psys.len());
    AccDot2Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.dot0[i][k];
            ftot1[k] += m * acc.dot1[i][k];
            ftot2[k] += m * acc.dot2[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("acc.dot0: {:?}", &acc.dot0[..2]);
    eprintln!("acc.dot1: {:?}", &acc.dot1[..2]);
    eprintln!("acc.dot2: {:?}", &acc.dot2[..2]);
    eprintln!("");

    psys.attrs.acc0 = acc.dot0;
    psys.attrs.acc1 = acc.dot1;
    let mut acc = Derivs3::zeros(psys.len());
    AccDot3Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    let mut ftot3 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.dot0[i][k];
            ftot1[k] += m * acc.dot1[i][k];
            ftot2[k] += m * acc.dot2[i][k];
            ftot3[k] += m * acc.dot3[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("ftot3: {:?}", ftot3);
    eprintln!("acc.dot0: {:?}", &acc.dot0[..2]);
    eprintln!("acc.dot1: {:?}", &acc.dot1[..2]);
    eprintln!("acc.dot2: {:?}", &acc.dot2[..2]);
    eprintln!("acc.dot3: {:?}", &acc.dot3[..2]);
    eprintln!("");

    // TODO: put this somewhere as a test.
    {
        let n = psys.len();
        let mut psys_1 = ParticleSystem::new();
        let mut psys_2 = ParticleSystem::new();
        let (attrs_1, attrs_2) = psys.attrs.split_at(n / 3);
        psys_1.attrs = attrs_1.to_vec();
        psys_2.attrs = attrs_2.to_vec();

        let (mtot_1, _rcom_1, vcom_1) = psys_1.com_mass_pos_vel();
        let vcom_1 = vcom_1.iter().fold(0.0, |s, v| s + v * v).sqrt();
        let kecom_1 = 0.5 * mtot_1 * vcom_1.powi(2);
        eprintln!("{:?} {:?}", mtot_1, kecom_1);

        let (mtot_2, _rcom_2, vcom_2) = psys_2.com_mass_pos_vel();
        let vcom_2 = vcom_2.iter().fold(0.0, |s, v| s + v * v).sqrt();
        let kecom_2 = 0.5 * mtot_2 * vcom_2.powi(2);
        eprintln!("{:?} {:?}", mtot_2, kecom_2);

        let mtot = psys.com_mass();
        // let mtot = mtot_1 + mtot_2;
        let kecom = kecom_1 + kecom_2;

        let mut energy = Energy::zeros(psys.len());
        EnergyKernel {}.compute(psys.attrs.as_slice().into(), energy.as_mut_slice());
        let (ke, pe) = energy.reduce(mtot);

        let mut energy_1 = Energy::zeros(psys_1.len());
        EnergyKernel {}.compute(psys_1.attrs.as_slice().into(), energy_1.as_mut_slice());
        let (ke_1, pe_1) = energy_1.reduce(mtot_1);

        let mut energy_2 = Energy::zeros(psys_2.len());
        EnergyKernel {}.compute(psys_2.attrs.as_slice().into(), energy_2.as_mut_slice());
        let (ke_2, pe_2) = energy_2.reduce(mtot_2);

        let mut energy_12 = Energy::zeros(psys_1.len());
        let mut energy_21 = Energy::zeros(psys_2.len());
        EnergyKernel {}.compute_mutual(
            psys_1.attrs.as_slice().into(),
            psys_2.attrs.as_slice().into(),
            energy_12.as_mut_slice(),
            energy_21.as_mut_slice(),
        );
        let (ke_12, pe_12) = energy_12.reduce(mtot);
        let (ke_21, pe_21) = energy_21.reduce(mtot);
        let ke_1221 = ke_12 + ke_21;
        let pe_1221 = pe_12 + pe_21;

        eprintln!("{:?} {:?}", ke, pe);
        eprintln!("{:?} {:?}", ke_1, pe_1);
        eprintln!("{:?} {:?}", ke_2, pe_2);
        eprintln!("{:?} {:?}", ke_1221, pe_1221);
        eprintln!("{:?} {:?}", ke_1 + ke_2 + kecom, pe_1221 + (pe_1 + pe_2));
        eprintln!(
            "{:?} {:?}",
            ke_1221 + (mtot_1 * ke_1 + mtot_2 * ke_2) / mtot,
            pe_1221 + (pe_1 + pe_2)
        );
    }

    for _ in 0..1 {
        let timer = Instant::now();
        let mut energy = Energy::zeros(psys.len());
        EnergyKernel {}.compute(psys.attrs.as_slice().into(), energy.as_mut_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("energies: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        let mut acc = Derivs0::zeros(psys.len());
        AccDot0Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc0: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        let mut acc = Derivs1::zeros(psys.len());
        AccDot1Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc1: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        let mut acc = Derivs2::zeros(psys.len());
        AccDot2Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc2: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        let mut acc = Derivs3::zeros(psys.len());
        AccDot3Kernel {}.compute(psys.attrs.as_slice().into(), acc.as_mut_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc3: {:?} {:.5?}", duration, ns_loop);
    }

    // --------------------

    let seed = [0; 32];
    let mut rng = StdRng::from_seed(seed);

    let npart = 256;
    // let imf = EqualMass::new(1.0);
    let imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    // let sdp = Dehnen0::new();
    let model = Model::new(imf, sdp);
    let psys =
        ParticleSystem::from_model(npart, &model, &mut rng).into_standard_units(0.5, Some(1.0));

    let eta = 0.5;

    let dtres_pow = -1;
    let dtlog_pow = -2;
    let dtmax_pow = -3;

    let tend = 10.0;

    use ava::sim::*;

    // let tstep_scheme = TimeStepScheme::constant(eta, dtres_pow, dtlog_pow, dtmax_pow);
    // let tstep_scheme = TimeStepScheme::variable(eta, dtres_pow, dtlog_pow, dtmax_pow);
    let tstep_scheme = TimeStepScheme::individual(eta, dtres_pow, dtlog_pow, dtmax_pow);

    // let integrator = Integrator::hermite4(1);
    // let integrator = Integrator::hermite6(1);
    let integrator = Integrator::hermite8(1);

    let mut instant = Instant::now();
    let mut sim = Simulation::new(psys, integrator, tstep_scheme, &mut instant);
    sim.evolve(tend, &mut instant)?;

    let file = File::open("res.sim")?;
    let mut reader = BufReader::new(file);
    let mut de_sim: Simulation = bincode::deserialize_from(&mut reader).unwrap();
    assert!(de_sim == sim);

    de_sim.evolve(tend, &mut instant)?;
    assert!(de_sim == sim);

    Ok(())
}

// -- end of file --
