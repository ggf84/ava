use real::Real;
use compute;
use sys::particles::Particle;
use super::{Integrator, TimeStepScheme::{self, *}};

fn to_power_of_two(dt: Real, dtmax: Real) -> Real {
    let pow = dt.log2().floor();
    let dtq = (2.0 as Real).powi(pow as i32);
    dtq.min(dtmax)
}

fn update_dtq(tnow: Real, dtold: Real, dtnew: Real, dtmax: Real) -> Real {
    if dtnew < dtold {
        0.5 * dtold
    } else {
        if dtnew > (2.0 * dtold) && tnow % (2.0 * dtold) == 0.0 {
            (2.0 * dtold).min(dtmax)
        } else {
            dtold
        }
    }
}

fn set_shared_dt(psys: &mut [Particle]) {
    let dt = psys[0].dt;
    for p in psys.iter_mut() {
        p.dt = dt;
    }
}

fn get_nact(tnew: Real, psys: &[Particle]) -> usize {
    let mut nact = 0;
    for p in psys.iter() {
        if p.tnow + p.dt == tnew {
            nact += 1
        } else {
            break;
        }
    }
    nact
}

trait Hermite: Integrator {
    const ORDER: u8;

    fn predict(&self, tnew: Real, psys: &mut [Particle]);
    fn evaluate(&self, nact: usize, psys: &mut [Particle]);
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]);
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle]);

    fn pec(
        &self,
        npec: u8,
        dtmax: Real,
        time_step_scheme: TimeStepScheme,
        psys: &mut [Particle],
    ) -> Real {
        match time_step_scheme {
            Constant { dt } => {
                psys[0].dt = to_power_of_two(dt, dtmax);
                set_shared_dt(&mut psys[..]);
            }
            Adaptive { shared } => {
                if shared {
                    set_shared_dt(&mut psys[..]);
                }
            }
        }
        let tnew = psys[0].tnow + psys[0].dt;
        let nact = get_nact(tnew, psys);
        let mut psys_new = psys.to_vec();
        self.predict(tnew, &mut psys_new[..]);
        for _ in 0..npec {
            self.evaluate(nact, &mut psys_new[..]);
            self.correct(&psys[..nact], &mut psys_new[..nact]);
        }
        self.interpolate(&psys[..nact], &mut psys_new[..nact]);
        psys_new[..nact].sort_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
        psys[..nact].clone_from_slice(&psys_new[..nact]);
        tnew
    }
}

pub struct Hermite4 {
    pub npec: u8,
    pub eta: Real,
    pub dtmax: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite4 {
    fn setup(&self, psys: &mut [Particle]) {
        // Init forces
        self.evaluate(psys.len(), &mut psys[..]);
        // Init time-steps
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let dt = 0.0625 * self.eta * (a0 / a1).sqrt();
            p.dt = to_power_of_two(dt, self.dtmax);
        }
        // Init sorting by time-steps
        psys.sort_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
    fn evolve(&self, psys: &mut [Particle]) -> Real {
        self.pec(self.npec, self.dtmax, self.time_step_scheme, psys)
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn evaluate(&self, nact: usize, psys: &mut [Particle]) {
        let (iiacc0, iiacc1) = compute::jrk::triangle(&psys[..nact]);
        let ((ijacc0, ijacc1), _) = compute::jrk::rectangle(&psys[..nact], &psys[nact..]);
        for (i, p) in psys[..nact].iter_mut().enumerate() {
            for k in 0..3 {
                p.acc0[k] = iiacc0[i][k] + ijacc0[i][k];
                p.acc1[k] = iiacc1[i][k] + ijacc1[i][k];
            }
        }
    }
    fn predict(&self, tnew: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);

            for k in 0..3 {
                let dpos = h3 * (p.acc1[k]);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                p.pos[k] += dpos;

                let dvel = h2 * (p.acc1[k]);
                let dvel = h1 * (p.acc0[k] + dvel);
                p.vel[k] += dvel;
            }
        }
    }
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;

            let d0 = c0;
            let d1 = c1 * (1.0 / 3.0);

            for k in 0..3 {
                let a0p = d0 * (pnew.acc0[k] + pold.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - pold.acc1[k]);
                pnew.vel[k] = pold.vel[k] + dt * (a0p - a1m);

                let v0p = d0 * (pnew.vel[k] + pold.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - pold.acc0[k]);
                pnew.pos[k] = pold.pos[k] + dt * (v0p - v1m);
            }
        }
    }
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
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
                // let a0p = c0 * (pnew.acc0[k] + pold.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - pold.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + pold.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - pold.acc1[k]);

                let a2mid = (1.0 / 2.0) * a1m;
                let a3mid = (1.0 / 2.0) * (a1p - a0m);

                let a2 = a2mid + 3.0 * a3mid;
                let a3 = a3mid;

                acc2[k] = s2 * a2;
                acc3[k] = s3 * a3;
            }
            pnew.tnow += dt;
            // Update time-steps
            let a0 = pnew.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = pnew.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();
            let dtnew = self.eta * (u / l).sqrt();
            pnew.dt = update_dtq(pnew.tnow, pnew.dt, dtnew, self.dtmax);
        }
    }
}

// --------------------

pub struct Hermite6 {
    pub npec: u8,
    pub eta: Real,
    pub dtmax: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite6 {
    fn setup(&self, psys: &mut [Particle]) {
        // Init forces
        self.evaluate(psys.len(), &mut psys[..]);
        self.evaluate(psys.len(), &mut psys[..]);
        // Init time-steps
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            // let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let dt = 0.0625 * self.eta * (a0 / a2).sqrt().sqrt();
            p.dt = to_power_of_two(dt, self.dtmax);
        }
        // Init sorting by time-steps
        psys.sort_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
    fn evolve(&self, psys: &mut [Particle]) -> Real {
        self.pec(self.npec, self.dtmax, self.time_step_scheme, psys)
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn evaluate(&self, nact: usize, psys: &mut [Particle]) {
        let (iiacc0, iiacc1, iiacc2) = compute::snp::triangle(&psys[..nact]);
        let ((ijacc0, ijacc1, ijacc2), _) = compute::snp::rectangle(&psys[..nact], &psys[nact..]);
        for (i, p) in psys[..nact].iter_mut().enumerate() {
            for k in 0..3 {
                p.acc0[k] = iiacc0[i][k] + ijacc0[i][k];
                p.acc1[k] = iiacc1[i][k] + ijacc1[i][k];
                p.acc2[k] = iiacc2[i][k] + ijacc2[i][k];
            }
        }
    }
    fn predict(&self, tnew: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);

            for k in 0..3 {
                let dpos = h5 * (p.acc3[k]);
                let dpos = h4 * (p.acc2[k] + dpos);
                let dpos = h3 * (p.acc1[k] + dpos);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                p.pos[k] += dpos;

                let dvel = h4 * (p.acc3[k]);
                let dvel = h3 * (p.acc2[k] + dvel);
                let dvel = h2 * (p.acc1[k] + dvel);
                let dvel = h1 * (p.acc0[k] + dvel);
                p.vel[k] += dvel;

                let dacc0 = h3 * (p.acc3[k]);
                let dacc0 = h2 * (p.acc2[k] + dacc0);
                let dacc0 = h1 * (p.acc1[k] + dacc0);
                p.acc0[k] += dacc0;
            }
        }
    }
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let d0 = c0;
            let d1 = c1 * (2.0 / 5.0);
            let d2 = c2 * (2.0 / 15.0);

            for k in 0..3 {
                let a0p = d0 * (pnew.acc0[k] + pold.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - pold.acc1[k]);
                let a2p = d2 * (pnew.acc2[k] + pold.acc2[k]);
                pnew.vel[k] = pold.vel[k] + dt * (a0p - (a1m - a2p));

                let v0p = d0 * (pnew.vel[k] + pold.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - pold.acc0[k]);
                let v2p = d2 * (pnew.acc1[k] + pold.acc1[k]);
                pnew.pos[k] = pold.pos[k] + dt * (v0p - (v1m - v2p));
            }
        }
    }
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
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
                // let a0p = c0 * (pnew.acc0[k] + pold.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - pold.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + pold.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - pold.acc1[k]);
                let a2p = c2 * (pnew.acc2[k] + pold.acc2[k]);
                let a2m = c2 * (pnew.acc2[k] - pold.acc2[k]);

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

                pnew.acc3[k] = acc3[k];
            }
            pnew.tnow += dt;
            // Update time-steps
            let a0 = pnew.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = pnew.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = pnew.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let a4 = acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = acc5.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            // let u = a2 + (a1 * a3).sqrt();
            // let u = a3 + (a2 * a4).sqrt();
            let l = a4 + (a3 * a5).sqrt();
            let dtnew = self.eta * (u / l).powf(1.0 / 6.0);
            pnew.dt = update_dtq(pnew.tnow, pnew.dt, dtnew, self.dtmax);
        }
    }
}

// --------------------

pub struct Hermite8 {
    pub npec: u8,
    pub eta: Real,
    pub dtmax: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite8 {
    fn setup(&self, psys: &mut [Particle]) {
        // Init forces
        self.evaluate(psys.len(), &mut psys[..]);
        self.evaluate(psys.len(), &mut psys[..]);
        // Init time-steps
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = p.acc3.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();
            let dt = 0.0625 * self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dt, self.dtmax);
        }
        // Init sorting by time-steps
        psys.sort_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
    fn evolve(&self, psys: &mut [Particle]) -> Real {
        self.pec(self.npec, self.dtmax, self.time_step_scheme, psys)
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn evaluate(&self, nact: usize, psys: &mut [Particle]) {
        let (iiacc0, iiacc1, iiacc2, iiacc3) = compute::crk::triangle(&psys[..nact]);
        let ((ijacc0, ijacc1, ijacc2, ijacc3), _) =
            compute::crk::rectangle(&psys[..nact], &psys[nact..]);
        for (i, p) in psys[..nact].iter_mut().enumerate() {
            for k in 0..3 {
                p.acc0[k] = iiacc0[i][k] + ijacc0[i][k];
                p.acc1[k] = iiacc1[i][k] + ijacc1[i][k];
                p.acc2[k] = iiacc2[i][k] + ijacc2[i][k];
                p.acc3[k] = iiacc3[i][k] + ijacc3[i][k];
            }
        }
    }
    fn predict(&self, tnew: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = tnew - p.tnow;
            let h1 = dt;
            let h2 = dt * (1.0 / 2.0);
            let h3 = dt * (1.0 / 3.0);
            let h4 = dt * (1.0 / 4.0);
            let h5 = dt * (1.0 / 5.0);
            let h6 = dt * (1.0 / 6.0);
            let h7 = dt * (1.0 / 7.0);

            for k in 0..3 {
                let dpos = h7 * (p.acc5[k]);
                let dpos = h6 * (p.acc4[k] + dpos);
                let dpos = h5 * (p.acc3[k] + dpos);
                let dpos = h4 * (p.acc2[k] + dpos);
                let dpos = h3 * (p.acc1[k] + dpos);
                let dpos = h2 * (p.acc0[k] + dpos);
                let dpos = h1 * (p.vel[k] + dpos);
                p.pos[k] += dpos;

                let dvel = h6 * (p.acc5[k]);
                let dvel = h5 * (p.acc4[k] + dvel);
                let dvel = h4 * (p.acc3[k] + dvel);
                let dvel = h3 * (p.acc2[k] + dvel);
                let dvel = h2 * (p.acc1[k] + dvel);
                let dvel = h1 * (p.acc0[k] + dvel);
                p.vel[k] += dvel;

                let dacc0 = h5 * (p.acc5[k]);
                let dacc0 = h4 * (p.acc4[k] + dacc0);
                let dacc0 = h3 * (p.acc3[k] + dacc0);
                let dacc0 = h2 * (p.acc2[k] + dacc0);
                let dacc0 = h1 * (p.acc1[k] + dacc0);
                p.acc0[k] += dacc0;

                let dacc1 = h4 * (p.acc5[k]);
                let dacc1 = h3 * (p.acc4[k] + dacc1);
                let dacc1 = h2 * (p.acc3[k] + dacc1);
                let dacc1 = h1 * (p.acc2[k] + dacc1);
                p.acc1[k] += dacc1;
            }
        }
    }
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
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
                let a0p = d0 * (pnew.acc0[k] + pold.acc0[k]);
                let a1m = d1 * (pnew.acc1[k] - pold.acc1[k]);
                let a2p = d2 * (pnew.acc2[k] + pold.acc2[k]);
                let a3m = d3 * (pnew.acc3[k] - pold.acc3[k]);
                pnew.vel[k] = pold.vel[k] + dt * (a0p - (a1m - (a2p - a3m)));

                let v0p = d0 * (pnew.vel[k] + pold.vel[k]);
                let v1m = d1 * (pnew.acc0[k] - pold.acc0[k]);
                let v2p = d2 * (pnew.acc1[k] + pold.acc1[k]);
                let v3m = d3 * (pnew.acc2[k] - pold.acc2[k]);
                pnew.pos[k] = pold.pos[k] + dt * (v0p - (v1m - (v2p - v3m)));
            }
        }
    }
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle]) {
        for (pold, pnew) in psys_old.iter().zip(psys_new.iter_mut()) {
            let dt = pnew.dt;
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
                // let a0p = c0 * (pnew.acc0[k] + pold.acc0[k]);
                let a0m = c0 * (pnew.acc0[k] - pold.acc0[k]);
                let a1p = c1 * (pnew.acc1[k] + pold.acc1[k]);
                let a1m = c1 * (pnew.acc1[k] - pold.acc1[k]);
                let a2p = c2 * (pnew.acc2[k] + pold.acc2[k]);
                let a2m = c2 * (pnew.acc2[k] - pold.acc2[k]);
                let a3p = c3 * (pnew.acc3[k] + pold.acc3[k]);
                let a3m = c3 * (pnew.acc3[k] - pold.acc3[k]);

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

                pnew.acc4[k] = acc4[k];
                pnew.acc5[k] = acc5[k];
            }
            pnew.tnow += dt;
            // Update time-steps
            let a0 = pnew.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = pnew.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = pnew.acc2.iter().fold(0.0, |s, v| s + v * v);
            // let a3 = pnew.acc3.iter().fold(0.0, |s, v| s + v * v);
            // let a4 = acc4.iter().fold(0.0, |s, v| s + v * v);
            let a5 = acc5.iter().fold(0.0, |s, v| s + v * v);
            let a6 = acc6.iter().fold(0.0, |s, v| s + v * v);
            let a7 = acc7.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            // let u = a2 + (a1 * a3).sqrt();
            // let u = a3 + (a2 * a4).sqrt();
            // let u = a4 + (a3 * a5).sqrt();
            // let u = a5 + (a4 * a6).sqrt();
            let l = a6 + (a5 * a7).sqrt();
            let dtnew = self.eta * (u / l).powf(1.0 / 10.0);
            pnew.dt = update_dtq(pnew.tnow, pnew.dt, dtnew, self.dtmax);
        }
    }
}

// -- end of file --
