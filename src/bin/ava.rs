use ava::{
    gravity::{Acc0, Acc1, Acc2, Acc3, Compute, Energy},
    ics::{
        imf::{EqualMass, Maschberger2013},
        sdp::{Dehnen0, Dehnen1, Dehnen12, Dehnen2, Dehnen32, Plummer},
        Model,
    },
    sys::ParticleSystem,
};
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
    for p in psys.attrs.iter() {
        writeln!(
            &mut writer,
            "{} {} {} \
             {} {} {} \
             {} {} {}",
            p.id, p.eps, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2],
        )?;
    }

    let psys1 =
        ParticleSystem::from_model(7 * npart, &model, &mut rng).into_standard_units(0.5, None);
    let psys2 =
        ParticleSystem::from_model(3 * npart, &model, &mut rng).into_standard_units(0.5, None);
    let (iacc_12, jacc_12) = Acc0 {}.compute_mutual(psys1.as_ref(), psys2.as_ref());
    let (iacc_21, jacc_21) = Acc0 {}.compute_mutual(psys2.as_ref(), psys1.as_ref());
    assert!(iacc_21 == jacc_12);
    assert!(iacc_12 == jacc_21);
    let mut ftot0 = [0.0; 3];
    for (i, m) in psys1.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * iacc_12.0[i][k];
        }
    }
    for (j, m) in psys2.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * jacc_12.0[j][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("iacc_12: {:?}", &iacc_12.0[..2]);
    eprintln!("jacc_21: {:?}", &jacc_21.0[..2]);
    eprintln!("jacc_12: {:?}", &jacc_12.0[..2]);
    eprintln!("iacc_21: {:?}", &iacc_21.0[..2]);
    eprintln!("");

    let acc = Acc0 {}.compute(psys.as_ref());
    let mut ftot0 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.0[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("acc.0: {:?}", &acc.0[..2]);
    eprintln!("");

    let acc = Acc1 {}.compute(psys.as_ref());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.0[i][k];
            ftot1[k] += m * acc.1[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("acc.0: {:?}", &acc.0[..2]);
    eprintln!("acc.1: {:?}", &acc.1[..2]);
    eprintln!("");

    psys.attrs.acc0 = acc.0;
    let acc = Acc2 {}.compute(psys.as_ref());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.0[i][k];
            ftot1[k] += m * acc.1[i][k];
            ftot2[k] += m * acc.2[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("acc.0: {:?}", &acc.0[..2]);
    eprintln!("acc.1: {:?}", &acc.1[..2]);
    eprintln!("acc.2: {:?}", &acc.2[..2]);
    eprintln!("");

    psys.attrs.acc0 = acc.0;
    psys.attrs.acc1 = acc.1;
    let acc = Acc3 {}.compute(psys.as_ref());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    let mut ftot3 = [0.0; 3];
    for (i, m) in psys.attrs.mass.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += m * acc.0[i][k];
            ftot1[k] += m * acc.1[i][k];
            ftot2[k] += m * acc.2[i][k];
            ftot3[k] += m * acc.3[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("ftot3: {:?}", ftot3);
    eprintln!("acc.0: {:?}", &acc.0[..2]);
    eprintln!("acc.1: {:?}", &acc.1[..2]);
    eprintln!("acc.2: {:?}", &acc.2[..2]);
    eprintln!("acc.3: {:?}", &acc.3[..2]);
    eprintln!("");

    // TODO: put this somewhere as a test.
    {
        let n = psys.len();
        let mut psys_1 = ParticleSystem::new();
        let mut psys_2 = ParticleSystem::new();
        psys_1.attrs = psys.attrs.slice(0..(n / 3)).to_vec();
        psys_2.attrs = psys.attrs.slice((n / 3)..n).to_vec();

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
        let (ke, pe) = Energy::new(mtot).energies(psys.as_ref());
        let (ke_1, pe_1) = Energy::new(mtot_1).energies(psys_1.as_ref());
        let (ke_2, pe_2) = Energy::new(mtot_2).energies(psys_2.as_ref());
        let (ke_12, pe_12) = Energy::new(mtot).energies_mutual(psys_1.as_ref(), psys_2.as_ref());
        eprintln!("{:?} {:?}", ke, pe);
        eprintln!("{:?} {:?}", ke_1, pe_1);
        eprintln!("{:?} {:?}", ke_2, pe_2);
        eprintln!("{:?} {:?}", ke_12, pe_12);
        eprintln!("{:?} {:?}", ke_1 + ke_2 + kecom, pe_12 + (pe_1 + pe_2));
        eprintln!(
            "{:?} {:?}",
            ke_12 + (mtot_1 * ke_1 + mtot_2 * ke_2) / mtot,
            pe_12 + (pe_1 + pe_2)
        );
    }

    for _ in 0..1 {
        let timer = Instant::now();
        // Pass mtot=1 because here we are not interested in the actual result.
        Energy::new(1.0).energies(psys.as_ref());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("energies: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc0 {}.compute(psys.as_ref());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc0: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc1 {}.compute(psys.as_ref());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc1: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc2 {}.compute(psys.as_ref());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc2: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc3 {}.compute(psys.as_ref());
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

    // use ava::real::Real;
    // let dt = eta * (2.0 as Real).powi(dtmax_pow);
    // let tstep_scheme = TimeStepScheme::constant(dt);
    // let tstep_scheme = TimeStepScheme::adaptive_shared();
    let tstep_scheme = TimeStepScheme::adaptive_block();

    // let integrator = Integrator::hermite4(tstep_scheme, eta, 1);
    // let integrator = Integrator::hermite6(tstep_scheme, eta, 1);
    let integrator = Integrator::hermite8(tstep_scheme, eta, 1);

    let mut instant = Instant::now();
    let mut sim = Simulation::new(
        psys,
        integrator,
        dtres_pow,
        dtlog_pow,
        dtmax_pow,
        &mut instant,
    );
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
