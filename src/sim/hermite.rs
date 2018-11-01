use super::{to_power_of_two, Counter, Evolver, TimeStepScheme};
use crate::{
    gravity::{
        acc0::{AccDot0Kernel, Derivs0, Derivs0Slice},
        acc1::{AccDot1Kernel, Derivs1, Derivs1Slice},
        acc2::{AccDot2Kernel, Derivs2, Derivs2Slice},
        acc3::{AccDot3Kernel, Derivs3, Derivs3Slice},
        Compute, SplitAt, SplitAtMut,
    },
    real::Real,
    sys::ParticleSystem,
};

use serde_derive::{Deserialize, Serialize};
use soa_derive::{soa_zip, soa_zip_impl};

fn count_nact(dt: Real, psys: &ParticleSystem) -> usize {
    // note: we assume psys has been sorted by dt.
    let mut nact = 0;
    let tnew = psys.time + dt;
    for (tnow, dt) in soa_zip!(&psys.attrs, [tnow, dt]) {
        if (tnow + dt) > tnew {
            break;
        } else {
            nact += 1
        }
    }
    nact
}

pub(super) trait Hermite {
    const ORDER: u8;
    fn npec(&self) -> u8;
    fn init_acc_dt(&self, nact: usize, psys: &mut ParticleSystem, eta: Real);
    fn predict(&self, nact: usize, psys: &mut ParticleSystem, dt: Real);
    fn ecorrect(&self, nact: usize, psys: &mut ParticleSystem);
    fn commit(&self, nact: usize, psys: &mut ParticleSystem, eta: Real);

    /// Set dt <= dtmax and sort by time-steps.
    fn set_dtmax_and_sort_by_dt(&self, nact: usize, psys: &mut ParticleSystem, mut dtmax: Real) {
        // ensures tnow is commensurable with dtmax
        let tnow = psys.time;
        while tnow % dtmax != 0.0 {
            dtmax *= 0.5;
        }

        // set time-step limits dt <= dtmax
        psys.attrs.dt[..nact]
            .iter_mut()
            .for_each(|dt| *dt = dtmax.min(*dt));

        // sort by dt
        let mut indices: Vec<_> = (0..nact).collect();
        indices.sort_by(|&i, &j| psys.attrs.dt[i].partial_cmp(&psys.attrs.dt[j]).unwrap());
        psys.attrs.apply_permutation(&indices);
    }
}

impl<T: Hermite> Evolver for T {
    fn init(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem) {
        let nact = psys.len();

        // init forces and time-steps
        self.init_acc_dt(nact, psys, tstep_scheme.eta);

        // set time-step limits and sort
        self.set_dtmax_and_sort_by_dt(nact, psys, tstep_scheme.dtmax);
    }

    fn evolve(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem) -> Counter {
        let mut counter = Counter::default();
        let tend = psys.time + tstep_scheme.dtmax;
        while psys.time < tend {
            let dt = tstep_scheme.match_dt(psys);
            let nact = count_nact(dt, psys);
            self.predict(psys.len(), psys, dt);
            for _ in 0..self.npec() {
                self.ecorrect(nact, psys);
            }
            self.commit(nact, psys, tstep_scheme.eta);
            self.set_dtmax_and_sort_by_dt(nact, psys, tstep_scheme.dtmax);
            counter.isteps += nact as u64;
            counter.bsteps += 1;
        }
        counter.steps += 1;
        counter
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite4 {
    npec: u8,
}
impl Hermite4 {
    pub fn new(npec: u8) -> Self {
        Hermite4 { npec }
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn npec(&self) -> u8 {
        self.npec
    }

    fn init_acc_dt(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        let mut acc = Derivs0::zeros(nact);
        AccDot0Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;

        // If all velocities are initially zero, then acc1 == 0. This means that
        // in order to have a robust value for dt we need to compute acc2 too.
        let mut acc = Derivs2::zeros(nact);
        AccDot2Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2]).take(nact)
        {
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);

            let u = a0;
            let l = a1 + (a0 * a2).sqrt();

            let dtr = 0.125 * eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }

    fn predict(&self, nact: usize, psys: &mut ParticleSystem, dt: Real) {
        psys.time += dt;

        for (&tnow, &pos, &vel, &acc0, &acc1, new_tnow, new_pos, new_vel) in soa_zip!(
            &mut psys.attrs,
            [tnow, pos, vel, acc0, acc1, mut new_tnow, mut new_pos, mut new_vel]
        )
        .take(nact)
        {
            let dt = psys.time - tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);

            *new_tnow = tnow + dt;
            for k in 0..3 {
                let dpos = h3 * (acc1[k]);
                let dpos = h2 * (acc0[k] + dpos);
                let dpos = h1 * (vel[k] + dpos);
                new_pos[k] = pos[k] + dpos;

                let dvel = h2 * (acc1[k]);
                let dvel = h1 * (acc0[k] + dvel);
                new_vel[k] = vel[k] + dvel;
            }
        }
    }

    fn ecorrect(&self, nact: usize, psys: &mut ParticleSystem) {
        let src = <AccDot1Kernel as Compute>::Input::new(
            &psys.attrs.eps,
            &psys.attrs.mass,
            Derivs1Slice(&psys.attrs.new_pos, &psys.attrs.new_vel),
        );
        let (src_lo, src_hi) = src.split_at(nact);

        let mut new_acc = Derivs1::zeros(psys.len());
        let mut dst = new_acc.as_mut_slice();
        let (mut dst_lo, mut dst_hi) = dst.split_at_mut(nact);

        let kernel = AccDot1Kernel {};
        kernel.compute(&src_lo, &mut dst_lo);
        kernel.compute_mutual(&src_lo, &src_hi, &mut dst_lo, &mut dst_hi);

        psys.attrs.new_acc0[..nact].copy_from_slice(&dst.0[..nact]);
        psys.attrs.new_acc1[..nact].copy_from_slice(&dst.1[..nact]);

        for (&dt, &pos, &vel, &acc0, &acc1, new_pos, new_vel, new_acc0, new_acc1) in soa_zip!(
            &mut psys.attrs,
            [dt, pos, vel, acc0, acc1, mut new_pos, mut new_vel, mut new_acc0, mut new_acc1]
        )
        .take(nact)
        {
            let h = dt * 0.5;

            let c0 = 0.5;
            let c1 = c0 * h;

            let d0 = c0;
            let d1 = c1 * (1.0 / 3.0);

            for k in 0..3 {
                let a0p = d0 * (new_acc0[k] + acc0[k]);
                let a1m = d1 * (new_acc1[k] - acc1[k]);
                new_vel[k] = vel[k] + dt * (a0p - a1m);

                let v0p = d0 * (new_vel[k] + vel[k]);
                let v1m = d1 * (new_acc0[k] - acc0[k]);
                new_pos[k] = pos[k] + dt * (v0p - v1m);
            }
        }
    }

    fn commit(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        for (dt, tnow, pos, vel, acc0, acc1, &new_tnow, &new_pos, &new_vel, &new_acc0, &new_acc1) in soa_zip!(
            &mut psys.attrs,
            [mut dt, mut tnow, mut pos, mut vel, mut acc0, mut acc1, new_tnow, new_pos, new_vel, new_acc0, new_acc1]
        )
        .take(nact)
        {
            let h = *dt * 0.5;
            let hinv = 1.0 / h;

            let c0 = 0.5;
            let c1 = c0 * h;

            let s1 = hinv;
            let s2 = s1 * (2.0 * hinv);
            let s3 = s2 * (3.0 * hinv);

            let mut _acc2 = [0.0; 3];
            let mut _acc3 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (new_acc0[k] + acc0[k]);
                let a0m = c0 * (new_acc0[k] - acc0[k]);
                let a1p = c1 * (new_acc1[k] + acc1[k]);
                let a1m = c1 * (new_acc1[k] - acc1[k]);

                let a2mid = (1.0 / 2.0) * a1m;
                let a3mid = (1.0 / 2.0) * (a1p - a0m);

                let a2 = a2mid + 3.0 * a3mid;
                let a3 = a3mid;

                _acc2[k] = s2 * a2;
                _acc3[k] = s3 * a3;
            }

            // Commit to the new state
            *tnow = new_tnow;
            *pos = new_pos;
            *vel = new_vel;
            *acc0 = new_acc0;
            *acc1 = new_acc1;

            // Update time-steps
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = _acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = _acc3.iter().fold(0.0, |s, v| s + v * v);

            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();

            let dtr = eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite6 {
    npec: u8,
}
impl Hermite6 {
    pub fn new(npec: u8) -> Self {
        Hermite6 { npec }
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn npec(&self) -> u8 {
        self.npec
    }

    fn init_acc_dt(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        let mut acc = Derivs0::zeros(nact);
        AccDot0Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;

        let mut acc = Derivs2::zeros(nact);
        AccDot2Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2]).take(nact)
        {
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = a1 * (a2 / a0); // acc3.iter().fold(0.0, |s, v| s + v * v);

            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();

            let dtr = 0.125 * eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }

    fn predict(&self, nact: usize, psys: &mut ParticleSystem, dt: Real) {
        psys.time += dt;

        for (&tnow, &pos, &vel, &acc0, &acc1, &acc2, &acc3, new_tnow, new_pos, new_vel, new_acc0) in soa_zip!(
            &mut psys.attrs,
            [tnow, pos, vel, acc0, acc1, acc2, acc3, mut new_tnow, mut new_pos, mut new_vel, mut new_acc0]
        ).take(nact) {
            let dt = psys.time - tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);

            *new_tnow = tnow + dt;
            for k in 0..3 {
                let dpos = h5 * (acc3[k]);
                let dpos = h4 * (acc2[k] + dpos);
                let dpos = h3 * (acc1[k] + dpos);
                let dpos = h2 * (acc0[k] + dpos);
                let dpos = h1 * (vel[k] + dpos);
                new_pos[k] = pos[k] + dpos;

                let dvel = h4 * (acc3[k]);
                let dvel = h3 * (acc2[k] + dvel);
                let dvel = h2 * (acc1[k] + dvel);
                let dvel = h1 * (acc0[k] + dvel);
                new_vel[k] = vel[k] + dvel;

                let dacc0 = h3 * (acc3[k]);
                let dacc0 = h2 * (acc2[k] + dacc0);
                let dacc0 = h1 * (acc1[k] + dacc0);
                new_acc0[k] = acc0[k] + dacc0;
            }
        }
    }

    fn ecorrect(&self, nact: usize, psys: &mut ParticleSystem) {
        let src = <AccDot2Kernel as Compute>::Input::new(
            &psys.attrs.eps,
            &psys.attrs.mass,
            Derivs2Slice(
                &psys.attrs.new_pos,
                &psys.attrs.new_vel,
                &psys.attrs.new_acc0,
            ),
        );
        let (src_lo, src_hi) = src.split_at(nact);

        let mut new_acc = Derivs2::zeros(psys.len());
        let mut dst = new_acc.as_mut_slice();
        let (mut dst_lo, mut dst_hi) = dst.split_at_mut(nact);

        let kernel = AccDot2Kernel {};
        kernel.compute(&src_lo, &mut dst_lo);
        kernel.compute_mutual(&src_lo, &src_hi, &mut dst_lo, &mut dst_hi);

        psys.attrs.new_acc0[..nact].copy_from_slice(&dst.0[..nact]);
        psys.attrs.new_acc1[..nact].copy_from_slice(&dst.1[..nact]);
        psys.attrs.new_acc2[..nact].copy_from_slice(&dst.2[..nact]);

        for (&dt, &pos, &vel, &acc0, &acc1, &acc2, new_pos, new_vel, new_acc0, new_acc1, new_acc2) in soa_zip!(
            &mut psys.attrs,
            [dt, pos, vel, acc0, acc1, acc2, mut new_pos, mut new_vel, mut new_acc0, mut new_acc1, mut new_acc2]
        )
        .take(nact)
        {
            let h = dt * 0.5;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let d0 = c0;
            let d1 = c1 * (2.0 / 5.0);
            let d2 = c2 * (2.0 / 15.0);

            for k in 0..3 {
                let a0p = d0 * (new_acc0[k] + acc0[k]);
                let a1m = d1 * (new_acc1[k] - acc1[k]);
                let a2p = d2 * (new_acc2[k] + acc2[k]);
                new_vel[k] = vel[k] + dt * (a0p - (a1m - a2p));

                let v0p = d0 * (new_vel[k] + vel[k]);
                let v1m = d1 * (new_acc0[k] - acc0[k]);
                let v2p = d2 * (new_acc1[k] + acc1[k]);
                new_pos[k] = pos[k] + dt * (v0p - (v1m - v2p));
            }
        }
    }

    fn commit(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        for (dt, tnow, pos, vel, acc0, acc1, acc2, acc3, &new_tnow, &new_pos, &new_vel, &new_acc0, &new_acc1, &new_acc2) in soa_zip!(
            &mut psys.attrs,
            [mut dt, mut tnow, mut pos, mut vel, mut acc0, mut acc1, mut acc2, mut acc3, new_tnow, new_pos, new_vel, new_acc0, new_acc1, new_acc2]
        )
        .take(nact)
        {
            let h = *dt * 0.5;
            let hinv = 1.0 / h;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let s1 = hinv;
            let s2 = s1 * (2.0 * hinv);
            let s3 = s2 * (3.0 * hinv);
            let s4 = s3 * (4.0 * hinv);
            let s5 = s4 * (5.0 * hinv);

            let mut _acc3 = [0.0; 3];
            let mut _acc4 = [0.0; 3];
            let mut _acc5 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (new_acc0[k] + acc0[k]);
                let a0m = c0 * (new_acc0[k] - acc0[k]);
                let a1p = c1 * (new_acc1[k] + acc1[k]);
                let a1m = c1 * (new_acc1[k] - acc1[k]);
                let a2p = c2 * (new_acc2[k] + acc2[k]);
                let a2m = c2 * (new_acc2[k] - acc2[k]);

                // even
                let tmp = a2p - (1.0 / 2.0) * a1m;
                let evn0 = (1.0 / 4.0) * tmp;
                let a4mid = evn0;

                // odd
                let tmp = a1p - a0m;
                let odd0 = (1.0 / 2.0) * tmp;
                let odd1 = (1.0 / 4.0) * a2m - (3.0 / 8.0) * tmp;
                let a3mid = odd0 - 2.0 * odd1;
                let a5mid = odd1;

                let a3 = a3mid + 4.0 * a4mid + 10.0 * a5mid;
                let a4 = a4mid + 5.0 * a5mid;
                let a5 = a5mid;

                _acc3[k] = s3 * a3;
                _acc4[k] = s4 * a4;
                _acc5[k] = s5 * a5;
            }

            // Commit to the new state
            *tnow = new_tnow;
            *pos = new_pos;
            *vel = new_vel;
            *acc0 = new_acc0;
            *acc1 = new_acc1;
            *acc2 = new_acc2;
            *acc3 = _acc3;

            // Update time-steps
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = _acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = _acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = _acc5.iter().fold(0.0, |s, v| s + v * v);
            // let u = a1 + (a0 * a2).sqrt();
            // let l = a4 + (a3 * a5).sqrt();
            // let dtr = eta * (u / l).powf(1.0 / 6.0);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();

            let u = b2 + (b1 * b3).sqrt();
            let l = b3 + (b2 * b4).sqrt();

            let dtr = eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite8 {
    npec: u8,
}
impl Hermite8 {
    pub fn new(npec: u8) -> Self {
        Hermite8 { npec }
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn npec(&self) -> u8 {
        self.npec
    }

    fn init_acc_dt(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        let mut acc = Derivs1::zeros(nact);
        AccDot1Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;

        let mut acc = Derivs3::zeros(nact);
        AccDot3Kernel {}.compute(&psys.as_slice(), &mut acc.as_mut_slice());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;
        psys.attrs.acc3 = acc.3;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2, &acc3) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2, acc3]).take(nact)
        {
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = a2 * (a2 / a0); // acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = a3 * (a2 / a0); // acc5.iter().fold(0.0, |s, v| s + v * v);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();

            let u = b2 + (b1 * b3).sqrt();
            let l = b3 + (b2 * b4).sqrt();

            let dtr = 0.125 * eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }

    fn predict(&self, nact: usize, psys: &mut ParticleSystem, dt: Real) {
        psys.time += dt;

        for (
            &tnow,
            &pos,
            &vel,
            &acc0,
            &acc1,
            &acc2,
            &acc3,
            &acc4,
            &acc5,
            new_tnow,
            new_pos,
            new_vel,
            new_acc0,
            new_acc1,
        ) in soa_zip!(
            &mut psys.attrs,
            [tnow, pos, vel, acc0, acc1, acc2, acc3, acc4, acc5, mut new_tnow, mut new_pos, mut new_vel, mut new_acc0, mut new_acc1]
        ).take(nact) {
            let dt = psys.time - tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);
            let h6 = dt * (1.0 / 6.0);
            let h7 = dt * (1.0 / 7.0);

            *new_tnow = tnow + dt;
            for k in 0..3 {
                let dpos = h7 * (acc5[k]);
                let dpos = h6 * (acc4[k] + dpos);
                let dpos = h5 * (acc3[k] + dpos);
                let dpos = h4 * (acc2[k] + dpos);
                let dpos = h3 * (acc1[k] + dpos);
                let dpos = h2 * (acc0[k] + dpos);
                let dpos = h1 * (vel[k] + dpos);
                new_pos[k] = pos[k] + dpos;

                let dvel = h6 * (acc5[k]);
                let dvel = h5 * (acc4[k] + dvel);
                let dvel = h4 * (acc3[k] + dvel);
                let dvel = h3 * (acc2[k] + dvel);
                let dvel = h2 * (acc1[k] + dvel);
                let dvel = h1 * (acc0[k] + dvel);
                new_vel[k] = vel[k] + dvel;

                let dacc0 = h5 * (acc5[k]);
                let dacc0 = h4 * (acc4[k] + dacc0);
                let dacc0 = h3 * (acc3[k] + dacc0);
                let dacc0 = h2 * (acc2[k] + dacc0);
                let dacc0 = h1 * (acc1[k] + dacc0);
                new_acc0[k] = acc0[k] + dacc0;

                let dacc1 = h4 * (acc5[k]);
                let dacc1 = h3 * (acc4[k] + dacc1);
                let dacc1 = h2 * (acc3[k] + dacc1);
                let dacc1 = h1 * (acc2[k] + dacc1);
                new_acc1[k] = acc1[k] + dacc1;
            }
        }
    }

    fn ecorrect(&self, nact: usize, psys: &mut ParticleSystem) {
        let src = <AccDot3Kernel as Compute>::Input::new(
            &psys.attrs.eps,
            &psys.attrs.mass,
            Derivs3Slice(
                &psys.attrs.new_pos,
                &psys.attrs.new_vel,
                &psys.attrs.new_acc0,
                &psys.attrs.new_acc1,
            ),
        );
        let (src_lo, src_hi) = src.split_at(nact);

        let mut new_acc = Derivs3::zeros(psys.len());
        let mut dst = new_acc.as_mut_slice();
        let (mut dst_lo, mut dst_hi) = dst.split_at_mut(nact);

        let kernel = AccDot3Kernel {};
        kernel.compute(&src_lo, &mut dst_lo);
        kernel.compute_mutual(&src_lo, &src_hi, &mut dst_lo, &mut dst_hi);

        psys.attrs.new_acc0[..nact].copy_from_slice(&dst.0[..nact]);
        psys.attrs.new_acc1[..nact].copy_from_slice(&dst.1[..nact]);
        psys.attrs.new_acc2[..nact].copy_from_slice(&dst.2[..nact]);
        psys.attrs.new_acc3[..nact].copy_from_slice(&dst.3[..nact]);

        for (&dt, &pos, &vel, &acc0, &acc1, &acc2, &acc3, new_pos, new_vel, new_acc0, new_acc1, new_acc2, new_acc3) in soa_zip!(
            &mut psys.attrs,
            [dt, pos, vel, acc0, acc1, acc2, acc3, mut new_pos, mut new_vel, mut new_acc0, mut new_acc1, mut new_acc2, mut new_acc3]
        )
        .take(nact)
        {
            let h = dt * 0.5;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);
            let c3 = c2 * h * (1.0 / 3.0);

            let d0 = c0;
            let d1 = c1 * (3.0 / 7.0);
            let d2 = c2 * (4.0 / 21.0);
            let d3 = c3 * (2.0 / 35.0);

            for k in 0..3 {
                let a0p = d0 * (new_acc0[k] + acc0[k]);
                let a1m = d1 * (new_acc1[k] - acc1[k]);
                let a2p = d2 * (new_acc2[k] + acc2[k]);
                let a3m = d3 * (new_acc3[k] - acc3[k]);
                new_vel[k] = vel[k] + dt * (a0p - (a1m - (a2p - a3m)));

                let v0p = d0 * (new_vel[k] + vel[k]);
                let v1m = d1 * (new_acc0[k] - acc0[k]);
                let v2p = d2 * (new_acc1[k] + acc1[k]);
                let v3m = d3 * (new_acc2[k] - acc2[k]);
                new_pos[k] = pos[k] + dt * (v0p - (v1m - (v2p - v3m)));
            }
        }
    }

    fn commit(&self, nact: usize, psys: &mut ParticleSystem, eta: Real) {
        for (dt, tnow, pos, vel, acc0, acc1, acc2, acc3, acc4, acc5, &new_tnow, &new_pos, &new_vel, &new_acc0, &new_acc1, &new_acc2, &new_acc3) in soa_zip!(
            &mut psys.attrs,
            [mut dt, mut tnow, mut pos, mut vel, mut acc0, mut acc1, mut acc2, mut acc3, mut acc4, mut acc5, new_tnow, new_pos, new_vel, new_acc0, new_acc1, new_acc2, new_acc3]
        )
        .take(nact)
        {
            let h = *dt * 0.5;
            let hinv = 1.0 / h;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);
            let c3 = c2 * h * (1.0 / 3.0);

            let s1 = hinv;
            let s2 = s1 * (2.0 * hinv);
            let s3 = s2 * (3.0 * hinv);
            let s4 = s3 * (4.0 * hinv);
            let s5 = s4 * (5.0 * hinv);
            let s6 = s5 * (6.0 * hinv);
            let s7 = s6 * (7.0 * hinv);

            let mut _acc4 = [0.0; 3];
            let mut _acc5 = [0.0; 3];
            let mut _acc6 = [0.0; 3];
            let mut _acc7 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (new_acc0[k] + acc0[k]);
                let a0m = c0 * (new_acc0[k] - acc0[k]);
                let a1p = c1 * (new_acc1[k] + acc1[k]);
                let a1m = c1 * (new_acc1[k] - acc1[k]);
                let a2p = c2 * (new_acc2[k] + acc2[k]);
                let a2m = c2 * (new_acc2[k] - acc2[k]);
                let a3p = c3 * (new_acc3[k] + acc3[k]);
                let a3m = c3 * (new_acc3[k] - acc3[k]);

                // even
                let tmp = a2p - (1.0 / 2.0) * a1m;
                let evn0 = (1.0 / 4.0) * tmp;
                let evn1 = (1.0 / 8.0) * (a3m - tmp);
                let a4mid = evn0 - 3.0 * evn1;
                let a6mid = evn1;

                // odd
                let tmp = a1p - a0m;
                let odd0 = (1.0 / 4.0) * (a2m - (3.0 / 2.0) * tmp);
                let odd1 = (1.0 / 8.0) * (a3p - 2.0 * a2m + (5.0 / 2.0) * tmp);
                let a5mid = odd0 - 3.0 * odd1;
                let a7mid = odd1;

                let a4 = a4mid + 5.0 * a5mid + 15.0 * a6mid + 35.0 * a7mid;
                let a5 = a5mid + 6.0 * a6mid + 21.0 * a7mid;
                let a6 = a6mid + 7.0 * a7mid;
                let a7 = a7mid;

                _acc4[k] = s4 * a4;
                _acc5[k] = s5 * a5;
                _acc6[k] = s6 * a6;
                _acc7[k] = s7 * a7;
            }

            // Commit to the new state
            *tnow = new_tnow;
            *pos = new_pos;
            *vel = new_vel;
            *acc0 = new_acc0;
            *acc1 = new_acc1;
            *acc2 = new_acc2;
            *acc3 = new_acc3;
            *acc4 = _acc4;
            *acc5 = _acc5;

            // Update time-steps
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = _acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = _acc5.iter().fold(0.0, |s, v| s + v * v);
            let a6 = _acc6.iter().fold(0.0, |s, v| s + v * v);
            let a7 = _acc7.iter().fold(0.0, |s, v| s + v * v);
            // let u = a1 + (a0 * a2).sqrt();
            // let l = a6 + (a5 * a7).sqrt();
            // let dtr = eta * (u / l).powf(1.0 / 10.0);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();
            let b5 = a5 + (a4 * a6).sqrt();
            let b6 = a6 + (a5 * a7).sqrt();

            let c2 = b2 + (b1 * b3).sqrt();
            let c3 = b3 + (b2 * b4).sqrt();
            let c4 = b4 + (b3 * b5).sqrt();
            let c5 = b5 + (b4 * b6).sqrt();

            let u = c3 + (c2 * c4).sqrt();
            let l = c4 + (c3 * c5).sqrt();

            let dtr = eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// -- end of file --
