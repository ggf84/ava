use real::Real;
use sys::system::ParticleSystem;
use std::time::{Duration, Instant};
use std::fmt::Debug;

pub mod hermite;

pub trait Integrator {
    fn init(&self, tnow: Real, psys: &mut ParticleSystem);
    fn evolve(&self, psys: &mut ParticleSystem, counter: &mut Counter) -> Real;
}

#[derive(Debug, Copy, Clone)]
pub enum TimeStepScheme {
    Constant { dt: Real },
    Adaptive { shared: bool },
}

#[derive(Debug, Default)]
pub struct Counter {
    bsteps: usize,
    isteps: usize,
}
impl Counter {
    fn reset(&mut self) {
        self.bsteps = 0;
        self.isteps = 0;
    }
}

#[derive(Debug, Default)]
pub struct Logger {
    te_0: Real,
    te_n: Real,
    counter: Counter,
    counter_tot: Counter,
    duration: Duration,
    duration_tot: Duration,
}
impl Logger {
    fn init(&mut self, psys: &ParticleSystem) {
        let ke = psys.kinectic_energy();
        let pe = psys.potential_energy();
        let te = ke + pe;
        self.te_0 = te;
        self.te_n = te;
        let h0 = format!(
            "# {:<12} {:<+11} {:<+11} {:<+11} {:<+11} {:<+11} \
             {:<+11} {:<+11} {:<+10} {:<+12} {:>6} {:>8} {:>10} {:>12}",
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
            "# {:<12} {:<+11} {:<+11} {:<+11} {:<+11} {:<+11} \
             {:<+11} {:<+11} {:<+10} {:<+12} {:>6} {:>8} {:>10} {:>12}",
            "tnow",
            "ke",
            "pe",
            "ve",
            "err_0",
            "err_n",
            "rcom",
            "vcom",
            "duration",
            "duration_tot",
            "bsteps",
            "isteps",
            "bsteps_tot",
            "isteps_tot",
        );
        println!("{}\n{}\n# {}", h0, h1, "-".repeat(h1.len() - 2));
    }
    fn log(&mut self, tnow: Real, psys: &ParticleSystem) {
        let rcom = psys.com_pos().iter().fold(0.0, |s, v| s + v * v).sqrt();
        let vcom = psys.com_vel().iter().fold(0.0, |s, v| s + v * v).sqrt();
        let ke = psys.kinectic_energy();
        let pe = psys.potential_energy();
        let te = ke + pe;
        let ve = 2.0 * ke + pe;
        let err_0 = (te - self.te_0) / self.te_0;
        let err_n = (te - self.te_n) / self.te_n;
        self.te_n = te;

        self.counter_tot.bsteps += self.counter.bsteps;
        self.counter_tot.isteps += self.counter.isteps;
        self.duration_tot += self.duration;

        let duration =
            self.duration.subsec_nanos() as f64 * 1.0e-9 + self.duration.as_secs() as f64;
        let duration_tot =
            self.duration_tot.subsec_nanos() as f64 * 1.0e-9 + self.duration_tot.as_secs() as f64;

        println!(
            "  {:<12.6e} {:<+11.4e} {:<+11.4e} {:<+11.4e} {:<+11.4e} {:<+11.4e} \
             {:<+11.4e} {:<+11.4e} {:<+10.4e} {:<+12.6e} {:>6} {:>8} {:>10} {:>12}",
            tnow,
            ke,
            pe,
            ve,
            err_0,
            err_n,
            rcom,
            vcom,
            duration,
            duration_tot,
            self.counter.bsteps,
            self.counter.isteps,
            self.counter_tot.bsteps,
            self.counter_tot.isteps,
        );
    }
}

#[derive(Debug)]
pub struct Simulation<I: Integrator + Debug> {
    tnow: Real,
    dtlog: Real,
    logger: Logger,
    instant: Instant,
    integrator: I,
}
impl<I: Integrator + Debug> Simulation<I> {
    pub fn new(integrator: I, dtlog: Real) -> Simulation<I> {
        Simulation {
            tnow: 0.0,
            dtlog: to_power_of_two(dtlog),
            logger: Default::default(),
            instant: Instant::now(),
            integrator: integrator,
        }
    }
    pub fn init(&mut self, tnow: Real, psys: &mut ParticleSystem) {
        self.tnow = tnow;
        self.logger.init(psys);
        self.integrator.init(tnow, psys);

        if self.tnow % self.dtlog == 0.0 {
            let instant = Instant::now();
            self.logger.duration = instant.duration_since(self.instant);
            self.logger.log(self.tnow, &psys);
            if self.tnow == tnow {
                eprintln!("{:#?}", self);
            }
            self.logger.counter.reset();
            self.instant = instant;
        }
    }
    pub fn run(&mut self, tend: Real, psys: &mut ParticleSystem) {
        while self.tnow < tend {
            self.tnow += self.integrator.evolve(psys, &mut self.logger.counter);

            if self.tnow % self.dtlog == 0.0 {
                let instant = Instant::now();
                self.logger.duration = instant.duration_since(self.instant);
                self.logger.log(self.tnow, &psys);
                if self.tnow == tend {
                    eprintln!("{:#?}", self);
                }
                self.logger.counter.reset();
                self.instant = instant;
            }
        }
    }
}

fn to_power_of_two(dt: Real) -> Real {
    let pow = dt.log2().floor();
    let dtq = (2.0 as Real).powi(pow as i32);
    dtq
}

// -- end of file --
