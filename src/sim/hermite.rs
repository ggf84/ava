use super::{to_power_of_two, Counter, Evolver, TimeStepScheme};
use crate::{
    gravity::{Acc0, Acc1, Acc2, Acc3, Compute},
    real::Real,
    sys::ParticleSystem,
};
use serde_derive::{Deserialize, Serialize};
use soa_derive::{soa_zip, soa_zip_impl};

fn count_nact(dt: Real, psys: &ParticleSystem) -> usize {
    let mut nact = 0;
    let tnew = psys.time + dt;
    for (tnow, dt) in soa_zip!(&psys.attrs, [tnow, dt]) {
        // note: assuming psys has been sorted by dt.
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
    fn tstep_scheme(&self) -> &TimeStepScheme;
    fn init_acc_dt(&self, psys: &mut ParticleSystem);
    fn predict(&self, dt: Real, psys: &mut ParticleSystem);
    fn ecorrect(&self, nact: usize, psys: &mut ParticleSystem);
    fn commit(&self, nact: usize, psys: &mut ParticleSystem);

    /// Set dt <= dtmax and sort by time-steps.
    fn set_dtmax_and_sort_by_dt(&self, mut dtmax: Real, nact: usize, psys: &mut ParticleSystem) {
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
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        // Init forces and time-steps
        self.init_acc_dt(psys);

        let nact = psys.len();
        self.set_dtmax_and_sort_by_dt(dtmax, nact, psys);
    }
    fn evolve(&self, dtmax: Real, psys: &mut ParticleSystem) -> Counter {
        let mut counter = Counter::new();
        let tend = psys.time + dtmax;
        while psys.time < tend {
            let dt = self.tstep_scheme().match_dt(psys);
            let nact = count_nact(dt, psys);
            self.predict(dt, psys);
            for _ in 0..self.npec() {
                self.ecorrect(nact, psys);
            }
            self.commit(nact, psys);
            self.set_dtmax_and_sort_by_dt(dtmax, nact, psys);
            counter.isteps += nact as u64;
            counter.bsteps += 1;
        }
        counter.steps += 1;
        counter
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite4 {
    tstep_scheme: TimeStepScheme,
    eta: Real,
    npec: u8,
}
impl Hermite4 {
    pub fn new(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Hermite4 {
            tstep_scheme,
            eta,
            npec,
        }
    }
}
impl Hermite for Hermite4 {
    const ORDER: u8 = 4;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> &TimeStepScheme {
        &self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let acc = Acc0 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        // If all velocities are initially zero, then acc1 == 0. This means that
        // in order to have a robust value for dt we need to compute acc2 too.
        let acc = Acc2 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2])
        {
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);

            let u = a0;
            let l = a1 + (a0 * a2).sqrt();

            let dtr = 0.125 * self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }
    fn predict(&self, dt: Real, psys: &mut ParticleSystem) {
        psys.time += dt;

        for (&tnow, &pos, &vel, &acc0, &acc1, new_tnow, new_pos, new_vel) in soa_zip!(
            &mut psys.attrs,
            [tnow, pos, vel, acc0, acc1, mut new_tnow, mut new_pos, mut new_vel]
        ) {
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
        use crate::gravity::acc1::SrcSlice;

        let kernel = Acc1 {};
        let new_acc_lo = kernel.compute(SrcSlice {
            eps: &psys.attrs.eps[..nact],
            mass: &psys.attrs.mass[..nact],
            rdot0: &psys.attrs.new_pos[..nact],
            rdot1: &psys.attrs.new_vel[..nact],
        });
        let (new_acc_hi, _) = kernel.compute_mutual(
            SrcSlice {
                eps: &psys.attrs.eps[..nact],
                mass: &psys.attrs.mass[..nact],
                rdot0: &psys.attrs.new_pos[..nact],
                rdot1: &psys.attrs.new_vel[..nact],
            },
            SrcSlice {
                eps: &psys.attrs.eps[nact..],
                mass: &psys.attrs.mass[nact..],
                rdot0: &psys.attrs.new_pos[nact..],
                rdot1: &psys.attrs.new_vel[nact..],
            },
        );
        psys.attrs.new_acc0[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.0.iter().flatten())
            .zip(new_acc_hi.0.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc1[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.1.iter().flatten())
            .zip(new_acc_hi.1.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });

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
    fn commit(&self, nact: usize, psys: &mut ParticleSystem) {
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

            let dtr = self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite6 {
    tstep_scheme: TimeStepScheme,
    eta: Real,
    npec: u8,
}
impl Hermite6 {
    pub fn new(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Hermite6 {
            tstep_scheme,
            eta,
            npec,
        }
    }
}
impl Hermite for Hermite6 {
    const ORDER: u8 = 6;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> &TimeStepScheme {
        &self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let acc = Acc0 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        let acc = Acc2 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2])
        {
            let a0 = acc0.iter().fold(0.0, |s, v| s + v * v);
            let a1 = acc1.iter().fold(0.0, |s, v| s + v * v);
            let a2 = acc2.iter().fold(0.0, |s, v| s + v * v);
            let a3 = a1 * (a2 / a0); // acc3.iter().fold(0.0, |s, v| s + v * v);

            let u = a1 + (a0 * a2).sqrt();
            let l = a2 + (a1 * a3).sqrt();

            let dtr = 0.125 * self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }
    fn predict(&self, dt: Real, psys: &mut ParticleSystem) {
        psys.time += dt;

        for (&tnow, &pos, &vel, &acc0, &acc1, &acc2, &acc3, new_tnow, new_pos, new_vel, new_acc0) in soa_zip!(
            &mut psys.attrs,
            [tnow, pos, vel, acc0, acc1, acc2, acc3, mut new_tnow, mut new_pos, mut new_vel, mut new_acc0]
        ) {
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
        use crate::gravity::acc2::SrcSlice;

        let kernel = Acc2 {};
        let new_acc_lo = kernel.compute(SrcSlice {
            eps: &psys.attrs.eps[..nact],
            mass: &psys.attrs.mass[..nact],
            rdot0: &psys.attrs.new_pos[..nact],
            rdot1: &psys.attrs.new_vel[..nact],
            rdot2: &psys.attrs.new_acc0[..nact],
        });
        let (new_acc_hi, _) = kernel.compute_mutual(
            SrcSlice {
                eps: &psys.attrs.eps[..nact],
                mass: &psys.attrs.mass[..nact],
                rdot0: &psys.attrs.new_pos[..nact],
                rdot1: &psys.attrs.new_vel[..nact],
                rdot2: &psys.attrs.new_acc0[..nact],
            },
            SrcSlice {
                eps: &psys.attrs.eps[nact..],
                mass: &psys.attrs.mass[nact..],
                rdot0: &psys.attrs.new_pos[nact..],
                rdot1: &psys.attrs.new_vel[nact..],
                rdot2: &psys.attrs.new_acc0[nact..],
            },
        );
        psys.attrs.new_acc0[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.0.iter().flatten())
            .zip(new_acc_hi.0.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc1[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.1.iter().flatten())
            .zip(new_acc_hi.1.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc2[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.2.iter().flatten())
            .zip(new_acc_hi.2.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });

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
    fn commit(&self, nact: usize, psys: &mut ParticleSystem) {
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
            // let dtr = self.eta * (u / l).powf(1.0 / 6.0);

            let b1 = a1 + (a0 * a2).sqrt();
            let b2 = a2 + (a1 * a3).sqrt();
            let b3 = a3 + (a2 * a4).sqrt();
            let b4 = a4 + (a3 * a5).sqrt();

            let u = b2 + (b1 * b3).sqrt();
            let l = b3 + (b2 * b4).sqrt();

            let dtr = self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// --------------------

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Hermite8 {
    tstep_scheme: TimeStepScheme,
    eta: Real,
    npec: u8,
}
impl Hermite8 {
    pub fn new(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Hermite8 {
            tstep_scheme,
            eta,
            npec,
        }
    }
}
impl Hermite for Hermite8 {
    const ORDER: u8 = 8;

    fn npec(&self) -> u8 {
        self.npec
    }
    fn tstep_scheme(&self) -> &TimeStepScheme {
        &self.tstep_scheme
    }
    fn init_acc_dt(&self, psys: &mut ParticleSystem) {
        let acc = Acc1 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        let acc = Acc3 {}.compute(psys.as_ref());
        psys.attrs.acc0 = acc.0;
        psys.attrs.acc1 = acc.1;
        psys.attrs.acc2 = acc.2;
        psys.attrs.acc3 = acc.3;

        let time = psys.time;

        for (dt, tnow, &acc0, &acc1, &acc2, &acc3) in
            soa_zip!(&mut psys.attrs, [mut dt, mut tnow, acc0, acc1, acc2, acc3])
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

            let dtr = 0.125 * self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
            *tnow = time;
        }
    }
    fn predict(&self, dt: Real, psys: &mut ParticleSystem) {
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
        ) {
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
        use crate::gravity::acc3::SrcSlice;

        let kernel = Acc3 {};
        let new_acc_lo = kernel.compute(SrcSlice {
            eps: &psys.attrs.eps[..nact],
            mass: &psys.attrs.mass[..nact],
            rdot0: &psys.attrs.new_pos[..nact],
            rdot1: &psys.attrs.new_vel[..nact],
            rdot2: &psys.attrs.new_acc0[..nact],
            rdot3: &psys.attrs.new_acc1[..nact],
        });
        let (new_acc_hi, _) = kernel.compute_mutual(
            SrcSlice {
                eps: &psys.attrs.eps[..nact],
                mass: &psys.attrs.mass[..nact],
                rdot0: &psys.attrs.new_pos[..nact],
                rdot1: &psys.attrs.new_vel[..nact],
                rdot2: &psys.attrs.new_acc0[..nact],
                rdot3: &psys.attrs.new_acc1[..nact],
            },
            SrcSlice {
                eps: &psys.attrs.eps[nact..],
                mass: &psys.attrs.mass[nact..],
                rdot0: &psys.attrs.new_pos[nact..],
                rdot1: &psys.attrs.new_vel[nact..],
                rdot2: &psys.attrs.new_acc0[nact..],
                rdot3: &psys.attrs.new_acc1[nact..],
            },
        );
        psys.attrs.new_acc0[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.0.iter().flatten())
            .zip(new_acc_hi.0.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc1[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.1.iter().flatten())
            .zip(new_acc_hi.1.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc2[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.2.iter().flatten())
            .zip(new_acc_hi.2.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });
        psys.attrs.new_acc3[..nact]
            .iter_mut()
            .flatten()
            .zip(new_acc_lo.3.iter().flatten())
            .zip(new_acc_hi.3.iter().flatten())
            .for_each(|((a, &a_lo), &a_hi)| {
                *a = a_lo + a_hi;
            });

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
    fn commit(&self, nact: usize, psys: &mut ParticleSystem) {
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
            // let dtr = self.eta * (u / l).powf(1.0 / 10.0);

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

            let dtr = self.eta * (u / l).sqrt();
            *dt = to_power_of_two(dtr);
        }
    }
}

// -- end of file --
