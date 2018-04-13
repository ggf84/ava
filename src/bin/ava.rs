extern crate ava;
extern crate rand;

use std::time::{Duration, Instant};
use rand::{SeedableRng, StdRng};

use ava::ics::Model;
use ava::ics::sdp::Plummer;
use ava::ics::imf::EqualMass;
use ava::ics::imf::Maschberger2013;

fn main() {
    let mut rng: StdRng = SeedableRng::from_seed(&[1, 2, 3, 4][..]);

    let _imf = EqualMass::new(1.0);
    let imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    let model = Model::new(imf, sdp);

    let mut ps = model.build(1024 * 8, &mut rng);

    /*
    for p in ps.particles.iter() {
        print!("{}\t{}", p.id, p.mass);
        print!("\t{}\t{}\t{}", p.pos[0], p.pos[1], p.pos[2]);
        print!("\t{}\t{}\t{}", p.vel[0], p.vel[1], p.vel[2]);
        print!("\t{}", p.eps2);
        println!();
    }
    */
    let ke = ps.kinectic_energy();
    let pe = ps.potential_energy();
    eprintln!("{:?} {:?} {:?}", ke, pe, ke + pe);
    eprintln!(
        "{:#?} {:#?} {:#?}",
        ps.com_mass(),
        ps.com_pos(),
        ps.com_vel()
    );
    eprintln!("");

    let ps1 = model.build(1024 * 11, &mut rng);
    let ps2 = model.build(1024 * 3, &mut rng);
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
        let now = Instant::now();
        ps.get_phi();
        // ps1.get_phi();
        // ps2.get_phi();
        // ps1.get_phi_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("phi: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_acc();
        // ps1.get_acc();
        // ps2.get_acc();
        // ps1.get_acc_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc0: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_jrk();
        // ps1.get_jrk();
        // ps2.get_jrk();
        // ps1.get_jrk_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc1: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_snp();
        // ps1.get_snp();
        // ps2.get_snp();
        // ps1.get_snp_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc2: {:?} {:.5?}", duration, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_crk();
        // ps1.get_crk();
        // ps2.get_crk();
        // ps1.get_crk_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        eprintln!("acc3: {:?} {:.5?}", duration, ns_loop);
    }

    // --------------------

    let mut rng: StdRng = SeedableRng::from_seed(&[1, 2, 3, 4][..]);

    let imf = EqualMass::new(1.0);
    let _imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    let model = Model::new(imf, sdp);

    let mut psys = model.build(256, &mut rng);

    let tend = 10.0;
    let mut tsim = 0.0;

    let eta = 0.75;

    let dtlog = 0.125;

    use ava::sim::{Integrator, TimeStepScheme::*};

    // use ava::sim::hermite::Hermite4;
    // let integrator = Hermite4 {
    //     npec: 2,
    //     eta: eta,
    //     // time_step_scheme: Constant,
    //     time_step_scheme: Adaptive { shared: true },
    // };

    // use ava::sim::hermite::Hermite6;
    // let integrator = Hermite6 {
    //     npec: 2,
    //     eta: eta,
    //     // time_step_scheme: Constant,
    //     time_step_scheme: Adaptive { shared: true },
    // };

    use ava::sim::hermite::Hermite8;
    let integrator = Hermite8 {
        npec: 2,
        eta: eta,
        // time_step_scheme: Constant,
        time_step_scheme: Adaptive { shared: true },
    };

    let now = Instant::now();
    let ke = psys.kinectic_energy();
    let pe = psys.potential_energy();
    let te_0 = ke + pe;
    let mut te_n = te_0;
    eprintln!("# system energy at t = {}: {:?}", tsim, te_0);
    integrator.setup(&mut psys);
    while tsim < tend {
        if tsim % dtlog == 0.0 {
            te_n = print_log(tsim, te_0, te_n, &psys, now.elapsed());
        }
        tsim += integrator.evolve(&mut psys);
    }
    if tsim % dtlog == 0.0 {
        te_n = print_log(tsim, te_0, te_n, &psys, now.elapsed());
    }
    eprintln!("# system energy at t = {}: {:?}", tsim, te_n);
    eprintln!("# total simulation time: {:?}", now.elapsed());
}

use ava::real::Real;
use ava::sys::system::ParticleSystem;
fn print_log(
    tsim: Real,
    te_0: Real,
    te_n: Real,
    psys: &ParticleSystem,
    duration: Duration,
) -> Real {
    let elapsed = duration.as_secs() as f64 + 1.0e-9 * f64::from(duration.subsec_nanos());
    let rcom = psys.com_pos().iter().fold(0.0, |s, v| s + v * v).sqrt();
    let vcom = psys.com_vel().iter().fold(0.0, |s, v| s + v * v).sqrt();
    let ke = psys.kinectic_energy();
    let pe = psys.potential_energy();
    let te = ke + pe;
    let ve = 2.0 * ke + pe;
    let err_0 = (te - te_0) / te_0;
    let err_n = (te - te_n) / te_n;
    println!(
        "{:<+12.5e} {:<+12.5e} {:<+12.5e} {:<+12.5e} {:<+12.5e} {:<+12.5e} {:<+12.5e} {:<+12.5e} {:<12.7e}",
        tsim, ke, pe, ve, err_0, err_n, rcom, vcom, elapsed
    );
    te
}

// -- end of file --
