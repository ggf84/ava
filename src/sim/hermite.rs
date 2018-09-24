use super::{to_power_of_two, Counter, Evolver, TimeStepScheme};
use crate::{
    gravity::{Acc1, Acc2, Acc3, Compute},
    real::Real,
    sys::ParticleSystem,
};
use serde_derive::{Deserialize, Serialize};

fn count_nact(tnew: Real, psys: &ParticleSystem) -> usize {
    let mut nact = 0;
    for p in psys.iter() {
        // note: assuming psys has been sorted by dt.
        if (p.tnow + p.dt) > tnew {
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
    fn tstep_scheme(&self) -> TimeStepScheme;
    fn init_acc_dt(&self, psys: &mut ParticleSystem);
    fn predict(&self, tnew: Real, psys: &ParticleSystem, psys_new: &mut ParticleSystem);
    fn ecorrect(&self, nact: usize, psys: &ParticleSystem, psys_new: &mut ParticleSystem);
    fn commit(&self, nact: usize, psys: &mut ParticleSystem, psys_new: &ParticleSystem);

    /// Set dt <= dtmax and sort by time-steps.
    fn set_dtmax_and_sort_by_dt(&self, nact: usize, psys: &mut ParticleSystem, mut dtmax: Real) {
        let (psys_lo, _) = psys.split_at_mut(nact);

        // ensures tnow is commensurable with dtmax
        let tnow = psys_lo[0].tnow;
        while tnow % dtmax != 0.0 {
            dtmax *= 0.5;
        }

        // set dt <= dtmax
        psys_lo.iter_mut().for_each(|p| p.dt = dtmax.min(p.dt));

        // sort by dt
        psys_lo.sort_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
}

impl<T: Hermite> Evolver for T {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        // Init forces and time-steps
        self.init_acc_dt(psys);

        let nact = psys.len();
        self.set_dtmax_and_sort_by_dt(nact, psys, dtmax);
    }
    fn evolve(&self, dtmax: Real, psys: &mut ParticleSystem) -> (Real, Counter) {
        let mut counter = Counter::new();
        let mut tnow = psys.particles[0].tnow;
        let tend = tnow + dtmax;
        while tnow < tend {
            let dt = self.tstep_scheme().match_dt(psys);
            let nact = count_nact(tnow + dt, psys);
            let mut psys_new = psys.clone();
            self.predict(tnow + dt, psys, &mut psys_new);
            for _ in 0..self.npec() {
                self.ecorrect(nact, psys, &mut psys_new);
            }
            self.commit(nact, psys, &psys_new);
            self.set_dtmax_and_sort_by_dt(nact, psys, dtmax);
            counter.isteps += nact as u64;
            counter.bsteps += 1;
            tnow += dt;
        }
        counter.steps += 1;
        (tnow, counter)
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite4 {
    eta: Real,
    tstep_scheme: TimeStepScheme,
    npec: u8,
    kernel: Acc1,
}
impl Hermite4 {
    pub fn new(eta: Real, tstep_scheme: TimeStepScheme, npec: u8) -> Self {
        Hermite4 {
            eta,
            tstep_scheme,
            npec,
            kernel: Acc1 {},
        }
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> TimeStepScheme {
        self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let iiacc = self.kernel.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
        }
        // If all velocities are initially zero, then acc1 == 0. This means that
        // in order to have a robust value for dt we need to compute acc2 too.
        let iiacc = Acc2 {}.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
            p.acc2 = iiacc.2[i];
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);

            let u = a0;
            let l = a1 + (a0 * a2).sqrt();

            let dt = 0.125 * self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dt);
        }
    }
    fn predict(&self, tnew: Real, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        for (p, pnew) in psys.iter().zip(psys_new.iter_mut()) {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);

            pnew.tnow = p.tnow + dt;
            for k in 0..3 {
                let dpos = h3 * (p.acc1[k]);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                pnew.pos[k] = p.pos[k] + dpos;

                let dvel = h2 * (p.acc1[k]);
                let dvel = h1 * (p.acc0[k] + dvel);
                pnew.vel[k] = p.vel[k] + dvel;
            }
        }
    }
    fn ecorrect(&self, nact: usize, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        let (psys_lo, _) = psys.split_at(nact);
        let (psys_new_lo, psys_new_hi) = psys_new.split_at_mut(nact);
        let acc_new_lo = self.kernel.compute(&psys_new_lo);
        let (acc_new_hi, _) = self.kernel.compute_mutual(&psys_new_lo, &psys_new_hi);
        for (i, (p, pnew)) in psys_lo.iter().zip(psys_new_lo.iter_mut()).enumerate() {
            let dt = p.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;

            let d0 = c0;
            let d1 = c1 * (1.0 / 3.0);

            for k in 0..3 {
                pnew.acc0[k] = acc_new_lo.0[i][k] + acc_new_hi.0[i][k];
                pnew.acc1[k] = acc_new_lo.1[i][k] + acc_new_hi.1[i][k];

                let a0p = d0 * (pnew.acc0[k] + p.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - p.acc1[k]);
                pnew.vel[k] = p.vel[k] + dt * (a0p - a1m);

                let v0p = d0 * (pnew.vel[k] + p.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - p.acc0[k]);
                pnew.pos[k] = p.pos[k] + dt * (v0p - v1m);
            }
        }
    }
    fn commit(&self, nact: usize, psys: &mut ParticleSystem, psys_new: &ParticleSystem) {
        let (psys_lo, _) = psys.split_at_mut(nact);
        let (psys_new_lo, _) = psys_new.split_at(nact);
        for (p, pnew) in psys_lo.iter_mut().zip(psys_new_lo.iter()) {
            let dt = p.dt;
            let h = 0.5 * dt;
            let hinv = 1.0 / h;

            let c0 = 0.5;
            let c1 = c0 * h;

            let s1 = hinv;
            let s2 = s1 * (2.0 * hinv);
            let s3 = s2 * (3.0 * hinv);

            let mut acc2 = [0.0; 3];
            let mut acc3 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (pnew.acc0[k] + p.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - p.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + p.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - p.acc1[k]);

                let a2mid = (1.0 / 2.0) * a1m;
                let a3mid = (1.0 / 2.0) * (a1p - a0m);

                let a2 = a2mid + 3.0 * a3mid;
                let a3 = a3mid;

                acc2[k] = s2 * a2;
                acc3[k] = s3 * a3;
            }

            // Commit to the new state
            p.tnow = pnew.tnow;
            p.pos = pnew.pos;
            p.vel = pnew.vel;
            p.acc0 = pnew.acc0;
            p.acc1 = pnew.acc1;

            // Update time-steps
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);

            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();

            let dtnew = self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dtnew);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite6 {
    eta: Real,
    tstep_scheme: TimeStepScheme,
    npec: u8,
    kernel: Acc2,
}
impl Hermite6 {
    pub fn new(eta: Real, tstep_scheme: TimeStepScheme, npec: u8) -> Self {
        Hermite6 {
            eta,
            tstep_scheme,
            npec,
            kernel: Acc2 {},
        }
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> TimeStepScheme {
        self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let iiacc = self.kernel.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
            p.acc2 = iiacc.2[i];
        }
        let iiacc = self.kernel.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
            p.acc2 = iiacc.2[i];
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = a1 * (a2 / a0); // p.acc3.iter().fold(0.0, |s, v| s + v * v);

            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();

            let dt = 0.125 * self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dt);
        }
    }
    fn predict(&self, tnew: Real, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        for (p, pnew) in psys.iter().zip(psys_new.iter_mut()) {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);

            pnew.tnow = p.tnow + dt;
            for k in 0..3 {
                let dpos = h5 * (p.acc3[k]);
                let dpos = h4 * (p.acc2[k] + dpos);
                let dpos = h3 * (p.acc1[k] + dpos);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                pnew.pos[k] = p.pos[k] + dpos;

                let dvel = h4 * (p.acc3[k]);
                let dvel = h3 * (p.acc2[k] + dvel);
                let dvel = h2 * (p.acc1[k] + dvel);
                let dvel = h1 * (p.acc0[k] + dvel);
                pnew.vel[k] = p.vel[k] + dvel;

                let dacc0 = h3 * (p.acc3[k]);
                let dacc0 = h2 * (p.acc2[k] + dacc0);
                let dacc0 = h1 * (p.acc1[k] + dacc0);
                pnew.acc0[k] = p.acc0[k] + dacc0;
            }
        }
    }
    fn ecorrect(&self, nact: usize, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        let (psys_lo, _) = psys.split_at(nact);
        let (psys_new_lo, psys_new_hi) = psys_new.split_at_mut(nact);
        let acc_new_lo = self.kernel.compute(&psys_new_lo);
        let (acc_new_hi, _) = self.kernel.compute_mutual(&psys_new_lo, &psys_new_hi);
        for (i, (p, pnew)) in psys_lo.iter().zip(psys_new_lo.iter_mut()).enumerate() {
            let dt = p.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let d0 = c0;
            let d1 = c1 * (2.0 / 5.0);
            let d2 = c2 * (2.0 / 15.0);

            for k in 0..3 {
                pnew.acc0[k] = acc_new_lo.0[i][k] + acc_new_hi.0[i][k];
                pnew.acc1[k] = acc_new_lo.1[i][k] + acc_new_hi.1[i][k];
                pnew.acc2[k] = acc_new_lo.2[i][k] + acc_new_hi.2[i][k];

                let a0p = d0 * (pnew.acc0[k] + p.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - p.acc1[k]);
                let a2p = d2 * (pnew.acc2[k] + p.acc2[k]);
                pnew.vel[k] = p.vel[k] + dt * (a0p - (a1m - a2p));

                let v0p = d0 * (pnew.vel[k] + p.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - p.acc0[k]);
                let v2p = d2 * (pnew.acc1[k] + p.acc1[k]);
                pnew.pos[k] = p.pos[k] + dt * (v0p - (v1m - v2p));
            }
        }
    }
    fn commit(&self, nact: usize, psys: &mut ParticleSystem, psys_new: &ParticleSystem) {
        let (psys_lo, _) = psys.split_at_mut(nact);
        let (psys_new_lo, _) = psys_new.split_at(nact);
        for (p, pnew) in psys_lo.iter_mut().zip(psys_new_lo.iter()) {
            let dt = p.dt;
            let h = 0.5 * dt;
            let hinv = 1.0 / h;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let s1 = hinv;
            let s2 = s1 * (2.0 * hinv);
            let s3 = s2 * (3.0 * hinv);
            let s4 = s3 * (4.0 * hinv);
            let s5 = s4 * (5.0 * hinv);

            let mut acc3 = [0.0; 3];
            let mut acc4 = [0.0; 3];
            let mut acc5 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (pnew.acc0[k] + p.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - p.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + p.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - p.acc1[k]);
                let a2p = c2 * (pnew.acc2[k] + p.acc2[k]);
                let a2m = c2 * (pnew.acc2[k] - p.acc2[k]);

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

                acc3[k] = s3 * a3;
                acc4[k] = s4 * a4;
                acc5[k] = s5 * a5;
            }

            // Commit to the new state
            p.tnow = pnew.tnow;
            p.pos = pnew.pos;
            p.vel = pnew.vel;
            p.acc0 = pnew.acc0;
            p.acc1 = pnew.acc1;
            p.acc2 = pnew.acc2;
            p.acc3 = acc3;

            // Update time-steps
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = acc5.iter().fold(0.0, |s, v| s + v * v);
            // let u = a1 + (a0 * a2).sqrt();
            // let l = a4 + (a3 * a5).sqrt();
            // let dtnew = self.eta * (u / l).powf(1.0 / 6.0);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();

            let u = b2 + (b1 * b3).sqrt();
            let l = b3 + (b2 * b4).sqrt();

            let dtnew = self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dtnew);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite8 {
    eta: Real,
    tstep_scheme: TimeStepScheme,
    npec: u8,
    kernel: Acc3,
}
impl Hermite8 {
    pub fn new(eta: Real, tstep_scheme: TimeStepScheme, npec: u8) -> Self {
        Hermite8 {
            eta,
            tstep_scheme,
            npec,
            kernel: Acc3 {},
        }
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> TimeStepScheme {
        self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let iiacc = self.kernel.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
            p.acc2 = iiacc.2[i];
            p.acc3 = iiacc.3[i];
        }
        let iiacc = self.kernel.compute(&psys.as_slice());
        for (i, p) in psys.iter_mut().enumerate() {
            p.acc0 = iiacc.0[i];
            p.acc1 = iiacc.1[i];
            p.acc2 = iiacc.2[i];
            p.acc3 = iiacc.3[i];
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = p.acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = a2 * (a2 / a0); // p.acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = a3 * (a2 / a0); // p.acc5.iter().fold(0.0, |s, v| s + v * v);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();

            let u = b2 + (b1 * b3).sqrt();
            let l = b3 + (b2 * b4).sqrt();

            let dt = 0.125 * self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dt);
        }
    }
    fn predict(&self, tnew: Real, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        for (p, pnew) in psys.iter().zip(psys_new.iter_mut()) {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);
            let h6 = dt * (1.0 / 6.0);
            let h7 = dt * (1.0 / 7.0);

            pnew.tnow = p.tnow + dt;
            for k in 0..3 {
                let dpos = h7 * (p.acc5[k]);
                let dpos = h6 * (p.acc4[k] + dpos);
                let dpos = h5 * (p.acc3[k] + dpos);
                let dpos = h4 * (p.acc2[k] + dpos);
                let dpos = h3 * (p.acc1[k] + dpos);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                pnew.pos[k] = p.pos[k] + dpos;

                let dvel = h6 * (p.acc5[k]);
                let dvel = h5 * (p.acc4[k] + dvel);
                let dvel = h4 * (p.acc3[k] + dvel);
                let dvel = h3 * (p.acc2[k] + dvel);
                let dvel = h2 * (p.acc1[k] + dvel);
                let dvel = h1 * (p.acc0[k] + dvel);
                pnew.vel[k] = p.vel[k] + dvel;

                let dacc0 = h5 * (p.acc5[k]);
                let dacc0 = h4 * (p.acc4[k] + dacc0);
                let dacc0 = h3 * (p.acc3[k] + dacc0);
                let dacc0 = h2 * (p.acc2[k] + dacc0);
                let dacc0 = h1 * (p.acc1[k] + dacc0);
                pnew.acc0[k] = p.acc0[k] + dacc0;

                let dacc1 = h4 * (p.acc5[k]);
                let dacc1 = h3 * (p.acc4[k] + dacc1);
                let dacc1 = h2 * (p.acc3[k] + dacc1);
                let dacc1 = h1 * (p.acc2[k] + dacc1);
                pnew.acc1[k] = p.acc1[k] + dacc1;
            }
        }
    }
    fn ecorrect(&self, nact: usize, psys: &ParticleSystem, psys_new: &mut ParticleSystem) {
        let (psys_lo, _) = psys.split_at(nact);
        let (psys_new_lo, psys_new_hi) = psys_new.split_at_mut(nact);
        let acc_new_lo = self.kernel.compute(&psys_new_lo);
        let (acc_new_hi, _) = self.kernel.compute_mutual(&psys_new_lo, &psys_new_hi);
        for (i, (p, pnew)) in psys_lo.iter().zip(psys_new_lo.iter_mut()).enumerate() {
            let dt = p.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);
            let c3 = c2 * h * (1.0 / 3.0);

            let d0 = c0;
            let d1 = c1 * (3.0 / 7.0);
            let d2 = c2 * (4.0 / 21.0);
            let d3 = c3 * (2.0 / 35.0);

            for k in 0..3 {
                pnew.acc0[k] = acc_new_lo.0[i][k] + acc_new_hi.0[i][k];
                pnew.acc1[k] = acc_new_lo.1[i][k] + acc_new_hi.1[i][k];
                pnew.acc2[k] = acc_new_lo.2[i][k] + acc_new_hi.2[i][k];
                pnew.acc3[k] = acc_new_lo.3[i][k] + acc_new_hi.3[i][k];

                let a0p = d0 * (pnew.acc0[k] + p.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - p.acc1[k]);
                let a2p = d2 * (pnew.acc2[k] + p.acc2[k]);
                let a3m = d3 * (pnew.acc3[k] - p.acc3[k]);
                pnew.vel[k] = p.vel[k] + dt * (a0p - (a1m - (a2p - a3m)));

                let v0p = d0 * (pnew.vel[k] + p.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - p.acc0[k]);
                let v2p = d2 * (pnew.acc1[k] + p.acc1[k]);
                let v3m = d3 * (pnew.acc2[k] - p.acc2[k]);
                pnew.pos[k] = p.pos[k] + dt * (v0p - (v1m - (v2p - v3m)));
            }
        }
    }
    fn commit(&self, nact: usize, psys: &mut ParticleSystem, psys_new: &ParticleSystem) {
        let (psys_lo, _) = psys.split_at_mut(nact);
        let (psys_new_lo, _) = psys_new.split_at(nact);
        for (p, pnew) in psys_lo.iter_mut().zip(psys_new_lo.iter()) {
            let dt = p.dt;
            let h = 0.5 * dt;
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

            let mut acc4 = [0.0; 3];
            let mut acc5 = [0.0; 3];
            let mut acc6 = [0.0; 3];
            let mut acc7 = [0.0; 3];
            for k in 0..3 {
                // let a0p = c0 * (pnew.acc0[k] + p.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - p.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + p.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - p.acc1[k]);
                let a2p = c2 * (pnew.acc2[k] + p.acc2[k]);
                let a2m = c2 * (pnew.acc2[k] - p.acc2[k]);
                let a3p = c3 * (pnew.acc3[k] + p.acc3[k]);
                let a3m = c3 * (pnew.acc3[k] - p.acc3[k]);

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

                acc4[k] = s4 * a4;
                acc5[k] = s5 * a5;
                acc6[k] = s6 * a6;
                acc7[k] = s7 * a7;
            }

            // Commit to the new state
            p.tnow = pnew.tnow;
            p.pos = pnew.pos;
            p.vel = pnew.vel;
            p.acc0 = pnew.acc0;
            p.acc1 = pnew.acc1;
            p.acc2 = pnew.acc2;
            p.acc3 = pnew.acc3;
            p.acc4 = acc4;
            p.acc5 = acc5;

            // Update time-steps
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = p.acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = acc5.iter().fold(0.0, |s, v| s + v * v);
            let a6 = acc6.iter().fold(0.0, |s, v| s + v * v);
            let a7 = acc7.iter().fold(0.0, |s, v| s + v * v);
            // let u = a1 + (a0 * a2).sqrt();
            // let l = a6 + (a5 * a7).sqrt();
            // let dtnew = self.eta * (u / l).powf(1.0 / 10.0);

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

            let dtnew = self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dtnew);
        }
    }
}

// -- end of file --
