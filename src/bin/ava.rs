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
        ParticleSystem::from_model(4 * npart, &model, &mut rng).to_standard_units(0.5, None);

    let file = File::create("cluster.txt")?;
    let mut writer = BufWriter::new(file);
    for p in psys.iter() {
        writeln!(
            &mut writer,
            "{} {} {} \
             {} {} {} \
             {} {} {}",
            p.id, p.eps, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2],
        )?;
    }

    let psys1 =
        ParticleSystem::from_model(7 * npart, &model, &mut rng).to_standard_units(0.5, None);
    let psys2 =
        ParticleSystem::from_model(3 * npart, &model, &mut rng).to_standard_units(0.5, None);
    let ((iacc0_21,), (jacc0_21,)) = Acc0 {}.compute_mutual(psys2.as_slice(), psys1.as_slice());
    let ((iacc0_12,), (jacc0_12,)) = Acc0 {}.compute_mutual(psys1.as_slice(), psys2.as_slice());
    assert!(iacc0_21 == jacc0_12);
    assert!(iacc0_12 == jacc0_21);
    let mut ftot0 = [0.0; 3];
    for (i, p) in psys1.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * iacc0_12[i][k];
        }
    }
    for (j, p) in psys2.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * jacc0_12[j][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("iacc0_12: {:?}", &iacc0_12[..2]);
    eprintln!("jacc0_21: {:?}", &jacc0_21[..2]);
    eprintln!("jacc0_12: {:?}", &jacc0_12[..2]);
    eprintln!("iacc0_21: {:?}", &iacc0_21[..2]);
    eprintln!("");

    let (acc0,) = Acc0 {}.compute(psys.as_slice());
    let mut ftot0 = [0.0; 3];
    for (i, p) in psys.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * acc0[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("_acc0: {:?}", &acc0[..2]);
    eprintln!("");

    let (acc0, acc1) = Acc1 {}.compute(psys.as_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    for (i, p) in psys.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * acc0[i][k];
            ftot1[k] += p.mass * acc1[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("_acc0: {:?}", &acc0[..2]);
    eprintln!("_acc1: {:?}", &acc1[..2]);
    eprintln!("");

    for (i, p) in psys.iter_mut().enumerate() {
        for k in 0..3 {
            p.acc0[k] = acc0[i][k];
        }
    }
    let (acc0, acc1, acc2) = Acc2 {}.compute(psys.as_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    for (i, p) in psys.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * acc0[i][k];
            ftot1[k] += p.mass * acc1[i][k];
            ftot2[k] += p.mass * acc2[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("_acc0: {:?}", &acc0[..2]);
    eprintln!("_acc1: {:?}", &acc1[..2]);
    eprintln!("_acc2: {:?}", &acc2[..2]);
    eprintln!("");

    for (i, p) in psys.iter_mut().enumerate() {
        for k in 0..3 {
            p.acc0[k] = acc0[i][k];
            p.acc1[k] = acc1[i][k];
        }
    }
    let (acc0, acc1, acc2, acc3) = Acc3 {}.compute(psys.as_slice());
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    let mut ftot3 = [0.0; 3];
    for (i, p) in psys.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * acc0[i][k];
            ftot1[k] += p.mass * acc1[i][k];
            ftot2[k] += p.mass * acc2[i][k];
            ftot3[k] += p.mass * acc3[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("ftot1: {:?}", ftot1);
    eprintln!("ftot2: {:?}", ftot2);
    eprintln!("ftot3: {:?}", ftot3);
    eprintln!("_acc0: {:?}", &acc0[..2]);
    eprintln!("_acc1: {:?}", &acc1[..2]);
    eprintln!("_acc2: {:?}", &acc2[..2]);
    eprintln!("_acc3: {:?}", &acc3[..2]);
    eprintln!("");

    // TODO: put this somewhere as a test.
    {
        let n = psys.len();
        let mut psys_1 = ParticleSystem::new();
        let mut psys_2 = ParticleSystem::new();
        psys_1.particles = psys.particles[..(n / 3)].to_vec();
        psys_2.particles = psys.particles[(n / 3)..].to_vec();

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
        let (ke, pe) = Energy::new(mtot).energies(psys.as_slice());
        let (ke_1, pe_1) = Energy::new(mtot_1).energies(psys_1.as_slice());
        let (ke_2, pe_2) = Energy::new(mtot_2).energies(psys_2.as_slice());
        let (ke_12, pe_12) =
            Energy::new(mtot).energies_mutual(psys_1.as_slice(), psys_2.as_slice());
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
        Energy::new(1.0).compute(psys.as_slice());
        // Energy::new(1.0).compute(psys1.as_slice());
        // Energy::new(1.0).compute(psys2.as_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("energies: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc0 {}.compute(psys.as_slice());
        // Acc0 {}.compute(psys1.as_slice());
        // Acc0 {}.compute(psys2.as_slice());
        // Acc0 {}.compute_mutual(psys1.as_slice(), psys2.as_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc0: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc1 {}.compute(psys.as_slice());
        // Acc1 {}.compute(psys1.as_slice());
        // Acc1 {}.compute(psys2.as_slice());
        // Acc1 {}.compute_mutual(psys1.as_slice(), psys2.as_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc1: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc2 {}.compute(psys.as_slice());
        // Acc2 {}.compute(psys1.as_slice());
        // Acc2 {}.compute(psys2.as_slice());
        // Acc2 {}.compute_mutual(psys1.as_slice(), psys2.as_slice());
        let duration = timer.elapsed();
        let elapsed = (1_000_000_000 * u128::from(duration.as_secs())
            + u128::from(duration.subsec_nanos())) as f64
            * 1.0e-9;
        let ns_loop = 1.0e9 * elapsed / (psys.len() as f64).powi(2);
        eprintln!("acc2: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        Acc3 {}.compute(psys.as_slice());
        // Acc3 {}.compute(psys1.as_slice());
        // Acc3 {}.compute(psys2.as_slice());
        // Acc3 {}.compute_mutual(psys1.as_slice(), psys2.as_slice());
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
        ParticleSystem::from_model(npart, &model, &mut rng).to_standard_units(0.5, Some(1.0));

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

    // let integrator = Hermite4::new(eta, tstep_scheme, 1);
    // let integrator = Hermite6::new(eta, tstep_scheme, 1);
    let integrator = Hermite8::new(eta, tstep_scheme, 1);

    let mut sim = Simulation::new(integrator, psys);
    sim.init(dtres_pow, dtlog_pow, dtmax_pow);
    sim.evolve(tend)?;

    let file = File::open("res.sim")?;
    let mut reader = BufReader::new(file);
    let mut de_sim: Simulation = bincode::deserialize_from(&mut reader).unwrap();
    assert!(de_sim == sim);

    de_sim.evolve(tend)?;
    assert!(de_sim == sim);

    Ok(())
}

// -- end of file --
