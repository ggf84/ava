extern crate ava;
extern crate rand;

use std::time::Instant;
use rand::{SeedableRng, StdRng};

use ava::ics::Model;
use ava::ics::sdp::Plummer;
use ava::ics::imf::{EqualMass, Maschberger2013};

fn main() {
    let mut rng: StdRng = SeedableRng::from_seed(&[1, 2, 3, 4][..]);

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

    let mut rng: StdRng = SeedableRng::from_seed(&[1, 2, 3, 4][..]);

    let imf = EqualMass::new(1.0);
    let _imf = Maschberger2013::new(0.01, 150.0);
    let sdp = Plummer::new();
    let model = Model::new(imf, sdp);

    let n = 256.0;
    let mut psys = model.build(n as usize, &mut rng);
    psys.set_eps(4.0 / n);

    let eta = 0.5;

    let dtmax = 0.0625;
    let dtlog = 0.125;

    let tnow = 0.0;
    let tend = 10.0;

    // use ava::sim::{TimeStepScheme::*, hermite::Hermite4};
    // let integrator = Hermite4::new(1, eta, dtmax, Constant { dt: eta * dtmax });
    // let integrator = Hermite4::new(1, eta, dtmax, Adaptive { shared: true });
    // let integrator = Hermite4::new(1, eta, dtmax, Adaptive { shared: false });

    // use ava::sim::{TimeStepScheme::*, hermite::Hermite6};
    // let integrator = Hermite6::new(1, eta, dtmax, Constant { dt: eta * dtmax });
    // let integrator = Hermite6::new(1, eta, dtmax, Adaptive { shared: true });
    // let integrator = Hermite6::new(1, eta, dtmax, Adaptive { shared: false });

    use ava::sim::{TimeStepScheme::*, hermite::Hermite8};
    // let integrator = Hermite8::new(1, eta, dtmax, Constant { dt: eta * dtmax });
    // let integrator = Hermite8::new(1, eta, dtmax, Adaptive { shared: true });
    let integrator = Hermite8::new(1, eta, dtmax, Adaptive { shared: false });

    use ava::sim::Simulation;
    let mut sim = Simulation::new(integrator, dtlog);
    sim.init(tnow, &mut psys);
    sim.run(tend, &mut psys);
}

// -- end of file --
