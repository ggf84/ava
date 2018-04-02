extern crate ava;
extern crate rand;

use std::time::Instant;
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

    let mut ps = model.build(1024 * 16, &mut rng);

    /*
    for p in ps.particles.iter() {
        print!("{}\t{}", p.id, p.m);
        print!("\t{}\t{}\t{}", p.r.0[0], p.r.0[1], p.r.0[2]);
        print!("\t{}\t{}\t{}", p.r.1[0], p.r.1[1], p.r.1[2]);
        print!("\t{}", p.e2);
        println!();
    }
    */
    let ke = ps.kinectic_energy();
    let pe = ps.potential_energy();
    eprintln!("{:?} {:?} {:?}", ke, pe, ke + pe);
    eprintln!("{:#?} {:#?} {:#?}", ps.com_m(), ps.com_r(), ps.com_v());

    let acc = ps.get_acc();
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].0[0];
        force_sum[1] += p.m * acc[i].0[1];
        force_sum[2] += p.m * acc[i].0[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    println!("_acc: {:?}", &acc[..2]);

    let ps1 = model.build(1024 * 13, &mut rng);
    let ps2 = model.build(1024 * 5, &mut rng);
    let (iacc, jacc) = ps1.get_acc_p2p(&ps2);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps1.particles.iter().enumerate() {
        force_sum[0] += p.m * iacc[i].0[0];
        force_sum[1] += p.m * iacc[i].0[1];
        force_sum[2] += p.m * iacc[i].0[2];
    }
    for (j, p) in ps2.particles.iter().enumerate() {
        force_sum[0] += p.m * jacc[j].0[0];
        force_sum[1] += p.m * jacc[j].0[1];
        force_sum[2] += p.m * jacc[j].0[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    println!("iacc: {:?}", &iacc[..2]);
    println!("jacc: {:?}", &jacc[..2]);

    let acc = ps.get_jrk();
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].0[0];
        force_sum[1] += p.m * acc[i].0[1];
        force_sum[2] += p.m * acc[i].0[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].1[0];
        force_sum[1] += p.m * acc[i].1[1];
        force_sum[2] += p.m * acc[i].1[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    println!("_acc: {:?}", &acc[..2]);

    for (i, p) in ps.particles.iter_mut().enumerate() {
        p.r.2[0] = acc[i].0[0];
        p.r.2[1] = acc[i].0[1];
        p.r.2[2] = acc[i].0[2];
    }
    let acc = ps.get_snp();
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].0[0];
        force_sum[1] += p.m * acc[i].0[1];
        force_sum[2] += p.m * acc[i].0[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].1[0];
        force_sum[1] += p.m * acc[i].1[1];
        force_sum[2] += p.m * acc[i].1[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].2[0];
        force_sum[1] += p.m * acc[i].2[1];
        force_sum[2] += p.m * acc[i].2[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    println!("_acc: {:?}", &acc[..2]);

    for (i, p) in ps.particles.iter_mut().enumerate() {
        p.r.2[0] = acc[i].0[0];
        p.r.2[1] = acc[i].0[1];
        p.r.2[2] = acc[i].0[2];
        p.r.3[0] = acc[i].1[0];
        p.r.3[1] = acc[i].1[1];
        p.r.3[2] = acc[i].1[2];
    }
    let acc = ps.get_crk();
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].0[0];
        force_sum[1] += p.m * acc[i].0[1];
        force_sum[2] += p.m * acc[i].0[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].1[0];
        force_sum[1] += p.m * acc[i].1[1];
        force_sum[2] += p.m * acc[i].1[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].2[0];
        force_sum[1] += p.m * acc[i].2[1];
        force_sum[2] += p.m * acc[i].2[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    let mut force_sum = [0.0; 3];
    for (i, p) in ps.particles.iter().enumerate() {
        force_sum[0] += p.m * acc[i].3[0];
        force_sum[1] += p.m * acc[i].3[1];
        force_sum[2] += p.m * acc[i].3[2];
    }
    println!("force_sum[0]: {:e}", force_sum[0]);
    println!("force_sum[1]: {:e}", force_sum[1]);
    println!("force_sum[2]: {:e}", force_sum[2]);
    println!("_acc: {:?}", &acc[..2]);

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_phi();
        // ps1.get_phi();
        // ps2.get_phi();
        // ps1.get_phi_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        println!("phi: {:>11} {:.5?}", elapsed, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_acc();
        // ps1.get_acc();
        // ps2.get_acc();
        // ps1.get_acc_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        println!("acc: {:>11} {:.5?}", elapsed, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_jrk();
        // ps1.get_jrk();
        // ps2.get_jrk();
        // ps1.get_jrk_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        println!("jrk: {:>11} {:.5?}", elapsed, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_snp();
        // ps1.get_snp();
        // ps2.get_snp();
        // ps1.get_snp_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        println!("snp: {:>11} {:.5?}", elapsed, ns_loop);
    }

    for _ in 0..1 {
        let now = Instant::now();
        ps.get_crk();
        // ps1.get_crk();
        // ps2.get_crk();
        // ps1.get_crk_p2p(&ps2);
        let duration = now.elapsed();
        let elapsed = u64::from(duration.subsec_nanos()) + 1_000_000_000 * duration.as_secs();
        let n = ps.particles.len() as f64;
        let ns_loop = elapsed as f64 / (n * n);
        println!("crk: {:>11} {:.5?}", elapsed, ns_loop);
    }
}

// -- end of file --
