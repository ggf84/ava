use real::Real;
use compute;
use sys::particles::Particle;
use sys::system::ParticleSystem;
use super::{Integrator, TimeStepScheme::{self, *}};

fn quantize_dt(dtr: Real, tnow: Real) -> Real {
    let mut dtq = 1.0;
    while dtq > dtr {
        dtq *= 0.5;
    }
    while (tnow + dtq) % dtq != 0.0 {
        dtq *= 0.5;
    }
    dtq
}

fn set_constant_dt(eta: Real, psys: &mut [Particle]) {
    let dtr = eta * 1.0;
    let dtq = quantize_dt(dtr, 0.0);
    for p in psys.iter_mut() {
        p.dt = dtq;
    }
}

fn set_shared_dt(psys: &mut [Particle]) {
    let dt = psys[0].dt;
    let dt = psys.iter().fold(dt, |dt, p| dt.min(p.dt));
    for p in psys.iter_mut() {
        p.dt = dt;
    }
}

trait Hermite: Integrator {
    const ORDER: u8;

    fn predict(&self, psys: &mut [Particle]);
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]);
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle]);

    fn pec(&self, npec: u8, psys: &mut ParticleSystem) -> Real {
        let dt = psys.particles[0].dt;
        let psys_old = psys.clone();
        self.predict(&mut psys.particles[..]);
        for _ in 0..npec {
            self.correct(&psys_old.particles[..], &mut psys.particles[..]);
        }
        self.interpolate(&psys_old.particles[..], &mut psys.particles[..]);
        dt
    }
}

pub struct Hermite4 {
    pub npec: u8,
    pub eta: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite4 {
    fn setup(&self, psys: &mut ParticleSystem) {
        // Init forces
        let (acc0, acc1) = compute::jrk::triangle(&psys.particles[..]);
        for (i, p) in psys.particles.iter_mut().enumerate() {
            for k in 0..3 {
                p.acc0[k] = acc0[i][k];
                p.acc1[k] = acc1[i][k];
            }
        }
        // Init time-steps
        for p in psys.particles.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let dtr = 0.125 * self.eta * (a0 / a1).sqrt();
            p.dt = quantize_dt(dtr, p.tnow);
        }
    }
    fn evolve(&self, psys: &mut ParticleSystem) -> Real {
        match self.time_step_scheme {
            Constant => {
                set_constant_dt(self.eta, &mut psys.particles[..]);
            }
            Adaptive { shared } => {
                if shared {
                    set_shared_dt(&mut psys.particles[..]);
                }
            }
        }
        self.pec(self.npec, psys)
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn predict(&self, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = p.dt;
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
        let (acc0, acc1) = compute::jrk::triangle(&psys_new[..]);
        for (i, (pold, pnew)) in psys_old.iter().zip(psys_new.iter_mut()).enumerate() {
            let dt = pnew.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;

            let d0 = c0;
            let d1 = c1 * (1.0 / 3.0);

            for k in 0..3 {
                pnew.acc0[k] = acc0[i][k];
                pnew.acc1[k] = acc1[i][k];

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

            pnew.tnow += dt;

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
            // Update time-steps
            let a0 = pnew.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = pnew.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = acc3.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();
            let dtr = self.eta * (u / l).sqrt();
            pnew.dt = quantize_dt(dtr, pnew.tnow);
        }
    }
}

// --------------------

pub struct Hermite6 {
    pub npec: u8,
    pub eta: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite6 {
    fn setup(&self, psys: &mut ParticleSystem) {
        // Init forces
        for _ in 0..2 {
            let (acc0, acc1, acc2) = compute::snp::triangle(&psys.particles[..]);
            for (i, p) in psys.particles.iter_mut().enumerate() {
                for k in 0..3 {
                    p.acc0[k] = acc0[i][k];
                    p.acc1[k] = acc1[i][k];
                    p.acc2[k] = acc2[i][k];
                }
            }
        }
        // Init time-steps
        for p in psys.particles.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            // let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let dtr = 0.125 * self.eta * (a0 / a2).sqrt().sqrt();
            p.dt = quantize_dt(dtr, p.tnow);
        }
    }
    fn evolve(&self, psys: &mut ParticleSystem) -> Real {
        match self.time_step_scheme {
            Constant => {
                set_constant_dt(self.eta, &mut psys.particles[..]);
            }
            Adaptive { shared } => {
                if shared {
                    set_shared_dt(&mut psys.particles[..]);
                }
            }
        }
        self.pec(self.npec, psys)
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn predict(&self, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = p.dt;
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
        let (acc0, acc1, acc2) = compute::snp::triangle(&psys_new[..]);
        for (i, (pold, pnew)) in psys_old.iter().zip(psys_new.iter_mut()).enumerate() {
            let dt = pnew.dt;
            let h = 0.5 * dt;

            let c0 = 0.5;
            let c1 = c0 * h;
            let c2 = c1 * h * (1.0 / 2.0);

            let d0 = c0;
            let d1 = c1 * (2.0 / 5.0);
            let d2 = c2 * (2.0 / 15.0);

            for k in 0..3 {
                pnew.acc0[k] = acc0[i][k];
                pnew.acc1[k] = acc1[i][k];
                pnew.acc2[k] = acc2[i][k];

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

            pnew.tnow += dt;

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
            let dtr = self.eta * (u / l).powf(1.0 / 6.0);
            pnew.dt = quantize_dt(dtr, pnew.tnow);
        }
    }
}

// --------------------

pub struct Hermite8 {
    pub npec: u8,
    pub eta: Real,
    pub time_step_scheme: TimeStepScheme,
}
impl Integrator for Hermite8 {
    fn setup(&self, psys: &mut ParticleSystem) {
        // Init forces
        for _ in 0..2 {
            let (acc0, acc1, acc2, acc3) = compute::crk::triangle(&psys.particles[..]);
            for (i, p) in psys.particles.iter_mut().enumerate() {
                for k in 0..3 {
                    p.acc0[k] = acc0[i][k];
                    p.acc1[k] = acc1[i][k];
                    p.acc2[k] = acc2[i][k];
                    p.acc3[k] = acc3[i][k];
                }
            }
        }
        // Init time-steps
        for p in psys.particles.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = p.acc3.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();
            let dtr = 0.125 * self.eta * (u / l).sqrt();
            p.dt = quantize_dt(dtr, p.tnow);
        }
    }
    fn evolve(&self, psys: &mut ParticleSystem) -> Real {
        match self.time_step_scheme {
            Constant => {
                set_constant_dt(self.eta, &mut psys.particles[..]);
            }
            Adaptive { shared } => {
                if shared {
                    set_shared_dt(&mut psys.particles[..]);
                }
            }
        }
        self.pec(self.npec, psys)
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn predict(&self, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let dt = p.dt;
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
        let (acc0, acc1, acc2, acc3) = compute::crk::triangle(&psys_new[..]);
        for (i, (pold, pnew)) in psys_old.iter().zip(psys_new.iter_mut()).enumerate() {
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
                pnew.acc0[k] = acc0[i][k];
                pnew.acc1[k] = acc1[i][k];
                pnew.acc2[k] = acc2[i][k];
                pnew.acc3[k] = acc3[i][k];

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

            pnew.tnow += dt;

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
            let dtr = self.eta * (u / l).powf(1.0 / 10.0);
            pnew.dt = quantize_dt(dtr, pnew.tnow);
        }
    }
}

// -- end of file --
