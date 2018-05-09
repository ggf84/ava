use real::Real;
use compute;
use sys::particles::Particle;
use sys::system::ParticleSystem;
use super::{to_power_of_two, Counter, Evolver, TimeStepScheme::{self, *}};

fn calc_dtlim(tnew: Real, dtmax: Real) -> Real {
    let mut dtlim = dtmax;
    while tnew % dtlim != 0.0 {
        dtlim *= 0.5;
    }
    dtlim
}

fn count_nact(tnew: Real, psys: &ParticleSystem) -> usize {
    let mut nact = 0;
    for p in psys.particles.iter() {
        if p.tnow + p.dt == tnew {
            nact += 1
        } else {
            break;
        }
    }
    nact
}

trait Hermite: Evolver {
    const ORDER: u8;

    fn init_dt(&self, dtmax: Real, psys: &mut [Particle]);
    fn evaluate(&self, nact: usize, psys: &mut [Particle]);
    fn predict(&self, tnew: Real, psys: &mut [Particle]);
    fn correct(&self, psys_old: &[Particle], psys_new: &mut [Particle]);
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle], dtlim: Real);

    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        let nact = psys.len();
        // Init forces
        self.evaluate(nact, &mut psys.particles[..]);
        if Self::ORDER > 4 {
            self.evaluate(nact, &mut psys.particles[..]);
        }
        // Init time-steps
        self.init_dt(dtmax, &mut psys.particles[..]);
        // Sort particles by time-steps
        psys.sort_by_dt(nact);
    }

    fn pec(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
        counter: &mut Counter,
        npec: u8,
    ) -> Real {
        let mut tnow = psys.particles[0].tnow;
        let dtmax = tend - tnow;
        while tnow < tend {
            match tstep_scheme {
                Constant { dt } => {
                    psys.set_shared_dt(dt);
                }
                Adaptive { shared } => {
                    if shared {
                        let dt = psys.particles[0].dt;
                        psys.set_shared_dt(dt);
                    }
                }
            }
            let dt = psys.particles[0].dt;
            let tnew = tnow + dt;
            let dtlim = calc_dtlim(tnew, dtmax);
            let nact = count_nact(tnew, psys);
            let mut psys_new = psys.clone();
            self.predict(tnew, &mut psys_new.particles[..]);
            for _ in 0..npec {
                self.evaluate(nact, &mut psys_new.particles[..]);
                self.correct(&psys.particles[..nact], &mut psys_new.particles[..nact]);
            }
            self.interpolate(
                &psys.particles[..nact],
                &mut psys_new.particles[..nact],
                dtlim,
            );
            psys.particles[..nact].clone_from_slice(&psys_new.particles[..nact]);
            psys.sort_by_dt(nact);
            counter.isteps += nact as u64;
            counter.bsteps += 1;
            tnow += dt;
        }
        counter.steps += 1;
        tnow
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite4 {
    eta: Real,
    npec: u8,
}
impl Hermite4 {
    pub fn new(eta: Real, npec: u8) -> Self {
        Hermite4 {
            eta: eta,
            npec: npec,
        }
    }
}
impl Evolver for Hermite4 {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        <Self as Hermite>::init(self, dtmax, psys);
    }
    fn evolve(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
        counter: &mut Counter,
    ) -> Real {
        self.pec(tend, psys, tstep_scheme, counter, self.npec)
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn init_dt(&self, dtmax: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let dt = 0.125 * self.eta * (a0 / a1).sqrt();
            p.dt = to_power_of_two(dt).min(dtmax);
        }
    }
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
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle], dtlim: Real) {
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
            pnew.dt = to_power_of_two(dtnew).min(dtlim);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite6 {
    eta: Real,
    npec: u8,
}
impl Hermite6 {
    pub fn new(eta: Real, npec: u8) -> Self {
        Hermite6 {
            eta: eta,
            npec: npec,
        }
    }
}
impl Evolver for Hermite6 {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        <Self as Hermite>::init(self, dtmax, psys);
    }
    fn evolve(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
        counter: &mut Counter,
    ) -> Real {
        self.pec(tend, psys, tstep_scheme, counter, self.npec)
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn init_dt(&self, dtmax: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            // let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let dt = 0.125 * self.eta * (a0 / a2).sqrt().sqrt();
            p.dt = to_power_of_two(dt).min(dtmax);
        }
    }
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
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle], dtlim: Real) {
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
            pnew.dt = to_power_of_two(dtnew).min(dtlim);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite8 {
    eta: Real,
    npec: u8,
}
impl Hermite8 {
    pub fn new(eta: Real, npec: u8) -> Self {
        Hermite8 {
            eta: eta,
            npec: npec,
        }
    }
}
impl Evolver for Hermite8 {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        <Self as Hermite>::init(self, dtmax, psys);
    }
    fn evolve(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
        counter: &mut Counter,
    ) -> Real {
        self.pec(tend, psys, tstep_scheme, counter, self.npec)
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn init_dt(&self, dtmax: Real, psys: &mut [Particle]) {
        for p in psys.iter_mut() {
            let a0 = p.acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = p.acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = p.acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = p.acc3.iter().fold(0.0, |s, v| s + v * v);
            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();
            let dt = 0.125 * self.eta * (u / l).sqrt();
            p.dt = to_power_of_two(dt).min(dtmax);
        }
    }
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
    fn interpolate(&self, psys_old: &[Particle], psys_new: &mut [Particle], dtlim: Real) {
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
            pnew.dt = to_power_of_two(dtnew).min(dtlim);
        }
    }
}

// -- end of file --
