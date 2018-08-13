extern crate ava;
extern crate bincode;
extern crate rand;

use rand::{SeedableRng, StdRng};
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use ava::ics::Model;
use ava::ics::imf::{EqualMass, Maschberger2013};
use ava::ics::sdp::Plummer;

fn main() {
    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let _imf = EqualMass::new(1.0);
    let imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    let model = Model::new(imf, sdp);

    let mut ps = model.build(1024 * 4, &mut rng);

    /*
    for p in ps.particles.iter() {
        print!("{}\t{}", p.id, p.mass);
        print!("\t{}\t{}\t{}", p.pos[0], p.pos[1], p.pos[2]);
        print!("\t{}\t{}\t{}", p.vel[0], p.vel[1], p.vel[2]);
        print!("\t{}", p.eps2);
        println!();
    }
    */
    let (ke, pe) = ps.energies();
    eprintln!("{:?} {:?} {:?}", ke, pe, ke + pe);
    eprintln!(
        "{:#?} {:#?} {:#?}",
        ps.com_mass(),
        ps.com_pos(),
        ps.com_vel()
    );
    eprintln!("");

    let ps1 = model.build(1024 * 7, &mut rng);
    let ps2 = model.build(1024 * 1, &mut rng);
    let ((iacc0,), (jacc0,)) = ps1.get_acc_p2p(&ps2);
    let mut ftot0 = [0.0; 3];
    for (i, p) in ps1.particles.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * iacc0[i][k];
        }
    }
    for (j, p) in ps2.particles.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * jacc0[j][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("iacc0: {:?}", &iacc0[..2]);
    eprintln!("jacc0: {:?}", &jacc0[..2]);
    eprintln!("");

    let (acc0,) = ps.get_acc();
    let mut ftot0 = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        for k in 0..3 {
            ftot0[k] += p.mass * acc0[i][k];
        }
    }
    eprintln!("ftot0: {:?}", ftot0);
    eprintln!("_acc0: {:?}", &acc0[..2]);
    eprintln!("");

    let (acc0, acc1) = ps.get_jrk();
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
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

    for (i, p) in ps.particles.iter_mut().enumerate() {
        for k in 0..3 {
            p.acc0[k] = acc0[i][k];
        }
    }
    let (acc0, acc1, acc2) = ps.get_snp();
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
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

    for (i, p) in ps.particles.iter_mut().enumerate() {
        for k in 0..3 {
            p.acc0[k] = acc0[i][k];
            p.acc1[k] = acc1[i][k];
        }
    }
    let (acc0, acc1, acc2, acc3) = ps.get_crk();
    let mut ftot0 = [0.0; 3];
    let mut ftot1 = [0.0; 3];
    let mut ftot2 = [0.0; 3];
    let mut ftot3 = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
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

    for _ in 0..1 {
        let timer = Instant::now();
        ps.energies();
        // ps1.energies();
        // ps2.energies();
        let duration = timer.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("energies: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        ps.get_acc();
        // ps1.get_acc();
        // ps2.get_acc();
        // ps1.get_acc_p2p(&ps2);
        let duration = timer.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc0: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        ps.get_jrk();
        // ps1.get_jrk();
        // ps2.get_jrk();
        // ps1.get_jrk_p2p(&ps2);
        let duration = timer.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc1: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        ps.get_snp();
        // ps1.get_snp();
        // ps2.get_snp();
        // ps1.get_snp_p2p(&ps2);
        let duration = timer.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc2: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let timer = Instant::now();
        ps.get_crk();
        // ps1.get_crk();
        // ps2.get_crk();
        // ps1.get_crk_p2p(&ps2);
        let duration = timer.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc3: {:?} {:.5?}", duration, ns_loop);
    }

    // --------------------

    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let imf = EqualMass::new(1.0);
    let _imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    let model = Model::new(imf, sdp);

    let n = 256.0;
    let mut psys = model.build(n as usize, &mut rng);
    psys.set_eps(4.0 / n);

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

    // let integrator = Hermite4::new(eta, 1);
    // let integrator = Hermite6::new(eta, 1);
    let integrator = Hermite8::new(eta, 1);

    let mut sim = Simulation::new(integrator, tstep_scheme, psys);
    sim.init(dtres_pow, dtlog_pow, dtmax_pow);
    sim.evolve(tend);

    let mut reader = BufReader::new(File::open("res.sim").unwrap());
    let mut de_sim: Simulation = bincode::deserialize_from(&mut reader).unwrap();
    assert!(de_sim == sim);

    de_sim.evolve(tend);
    assert!(de_sim == sim);
}

// -- end of file --
