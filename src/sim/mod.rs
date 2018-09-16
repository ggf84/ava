pub mod hermite;

pub use self::hermite::{Hermite4, Hermite6, Hermite8};
use crate::{real::Real, sys::system::ParticleSystem};
use serde_derive::{Deserialize, Serialize};
use std::{
    fs::File,
    io::BufWriter,
    ops::{Add, AddAssign},
    time::{Duration, Instant},
};

fn to_power_of_two(dt: Real) -> Real {
    let pow = dt.log2().floor();
    (2.0 as Real).powi(pow as i32)
}

pub(crate) trait Evolver {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem);
    fn evolve(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
    ) -> (Real, Counter);
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Counter {
    steps: u16,
    bsteps: u32,
    isteps: u64,
}
impl Counter {
    fn new() -> Self {
        Default::default()
    }
}
impl Add for Counter {
    type Output = Counter;
    fn add(self, other: Counter) -> Counter {
        Counter {
            steps: self.steps + other.steps,
            bsteps: self.bsteps + other.bsteps,
            isteps: self.isteps + other.isteps,
        }
    }
}
impl AddAssign for Counter {
    fn add_assign(&mut self, other: Counter) {
        *self = *self + other;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TimeStepScheme {
    Constant { dt: Real },
    Adaptive { shared: bool },
}
impl TimeStepScheme {
    pub fn constant(dt: Real) -> TimeStepScheme {
        TimeStepScheme::Constant {
            dt: to_power_of_two(dt),
        }
    }
    pub fn adaptive_shared() -> TimeStepScheme {
        TimeStepScheme::Adaptive { shared: true }
    }
    pub fn adaptive_block() -> TimeStepScheme {
        TimeStepScheme::Adaptive { shared: false }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum IntegratorKind {
    H4(Hermite4),
    H6(Hermite6),
    H8(Hermite8),
}
impl From<Hermite4> for IntegratorKind {
    fn from(integrator: Hermite4) -> Self {
        IntegratorKind::H4(integrator)
    }
}
impl From<Hermite6> for IntegratorKind {
    fn from(integrator: Hermite6) -> Self {
        IntegratorKind::H6(integrator)
    }
}
impl From<Hermite8> for IntegratorKind {
    fn from(integrator: Hermite8) -> Self {
        IntegratorKind::H8(integrator)
    }
}
impl Evolver for IntegratorKind {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem) {
        match self {
            IntegratorKind::H4(integrator) => integrator.init(dtmax, psys),
            IntegratorKind::H6(integrator) => integrator.init(dtmax, psys),
            IntegratorKind::H8(integrator) => integrator.init(dtmax, psys),
        }
    }
    fn evolve(
        &self,
        tend: Real,
        psys: &mut ParticleSystem,
        tstep_scheme: TimeStepScheme,
    ) -> (Real, Counter) {
        match self {
            IntegratorKind::H4(integrator) => integrator.evolve(tend, psys, tstep_scheme),
            IntegratorKind::H6(integrator) => integrator.evolve(tend, psys, tstep_scheme),
            IntegratorKind::H8(integrator) => integrator.evolve(tend, psys, tstep_scheme),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Simulation {
    tnow: Real,
    dtres: Real,
    dtlog: Real,
    dtmax: Real,
    logger: Logger,
    tstep_scheme: TimeStepScheme,
    integrator: IntegratorKind,
    psys: ParticleSystem,
}
impl Simulation {
    pub fn new<I: Into<IntegratorKind>>(
        integrator: I,
        tstep_scheme: TimeStepScheme,
        psys: ParticleSystem,
    ) -> Simulation {
        Simulation {
            tnow: 0.0,
            dtres: 0.0,
            dtlog: 0.0,
            dtmax: 0.0,
            logger: Default::default(),
            integrator: integrator.into(),
            tstep_scheme,
            psys,
        }
    }
    fn write_restart_file(&self) -> Result<(), ::std::io::Error> {
        let file = File::create("res.sim")?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &self).unwrap();
        Ok(())
    }
}
impl Simulation {
    pub fn init(&mut self, dtres_pow: i32, dtlog_pow: i32, dtmax_pow: i32) {
        let mut instant = Instant::now();
        assert!(dtres_pow >= dtlog_pow);
        assert!(dtlog_pow >= dtmax_pow);
        self.dtres = (2.0 as Real).powi(dtres_pow);
        self.dtlog = (2.0 as Real).powi(dtlog_pow);
        self.dtmax = (2.0 as Real).powi(dtmax_pow);
        self.integrator.init(self.dtmax, &mut self.psys);
        self.logger.init(&self.psys);
        self.logger.log(self.tnow, &self.psys, &mut instant);
    }
    pub fn evolve(&mut self, tend: Real) -> Result<(), ::std::io::Error> {
        let mut instant = Instant::now();
        while self.tnow < tend {
            let tend = self.tnow + self.dtmax;
            let (tnow, counter) = self
                .integrator
                .evolve(tend, &mut self.psys, self.tstep_scheme);
            self.tnow = tnow;
            self.logger.counter += counter;

            if self.tnow % self.dtlog == 0.0 {
                self.logger.log(self.tnow, &self.psys, &mut instant);
            }
            if self.tnow % self.dtres == 0.0 {
                self.write_restart_file()?;
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug, PartialEq, Serialize, Deserialize)]
struct Logger {
    te_0: Real,
    te_n: Real,
    counter: Counter,
    counter_cum: Counter,
    duration: Duration,
    duration_cum: Duration,
}
impl Logger {
    fn init(&mut self, psys: &ParticleSystem) {
        let (ke, pe) = psys.energies();
        let te = ke + pe;
        self.te_0 = te;
        self.te_n = te;
        let h0 = format!(
            "{:>11} {:>+10} {:>+10} {:>+10} \
             {:>+10} {:>+10} {:>+10} {:>+10} \
             {:>6} {:>8} {:>10} {:>12} {:>9} {:>12}",
            "(1)",
            "(2)",
            "(3)",
            "(4)",
            "(5)",
            "(6)",
            "(7)",
            "(8)",
            "(9)",
            "(10)",
            "(11)",
            "(12)",
            "(13)",
            "(14)",
        );
        let h1 = format!(
            "{:>11} {:>+10} {:>+10} {:>+10} \
             {:>+10} {:>+10} {:>+10} {:>+10} \
             {:>6} {:>8} {:>10} {:>12} {:>9} {:>12}",
            "tnow",
            "ke",
            "pe",
            "ve",
            "err_rel",
            "err_cum",
            "rcom",
            "vcom",
            "bsteps",
            "isteps",
            "bsteps_cum",
            "isteps_cum",
            "duration",
            "duration_cum",
        );
        println!("# {}\n# {}\n# {}", h0, h1, "-".repeat(h1.len()));
    }
    fn log(&mut self, tnow: Real, psys: &ParticleSystem, instant: &mut Instant) {
        let (_, rcom, vcom) = psys.com_mass_pos_vel();
        let rcom = rcom.iter().fold(0.0, |s, r| s + r * r).sqrt();
        let vcom = vcom.iter().fold(0.0, |s, v| s + v * v).sqrt();
        let (ke, pe) = psys.energies();
        let te = ke + pe;
        let ve = 2.0 * ke + pe;
        let err_rel = (te - self.te_n) / self.te_n;
        let err_cum = (te - self.te_0) / self.te_0;
        self.te_n = te;

        let now = Instant::now();
        self.duration = now.duration_since(*instant);

        self.counter_cum += self.counter;
        self.duration_cum += self.duration;

        let duration = (1_000_000_000 * u128::from(self.duration.as_secs())
            + u128::from(self.duration.subsec_nanos())) as f64
            * 1.0e-9;
        let duration_cum = (1_000_000_000 * u128::from(self.duration_cum.as_secs())
            + u128::from(self.duration_cum.subsec_nanos())) as f64
            * 1.0e-9;

        let line = format!(
            "{:>11.4} {:>+10.3e} {:>+10.3e} {:>+10.3e} \
             {:>+10.3e} {:>+10.3e} {:>+10.3e} {:>+10.3e} \
             {:>6} {:>8} {:>10} {:>12} {:>9.4} {:>12.4}",
            tnow,
            ke,
            pe,
            ve,
            err_rel,
            err_cum,
            rcom,
            vcom,
            self.counter.bsteps,
            self.counter.isteps,
            self.counter_cum.bsteps,
            self.counter_cum.isteps,
            duration,
            duration_cum,
        );
        println!("  {}", line);
        self.counter = Counter::new();
        *instant = now;
    }
}

// -- end of file --
