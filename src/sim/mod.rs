use real::Real;
use sys::particles::Particle;
use sys::system::ParticleSystem;
use std::time::{Duration, Instant};
use std::fmt::Debug;

pub mod hermite;

pub trait Integrator {
    fn init(&self, tnow: Real, psys: &mut [Particle]);
    fn evolve(&self, psys: &mut [Particle]) -> Real;
}

#[derive(Debug, Copy, Clone)]
pub enum TimeStepScheme {
    Constant { dt: Real },
    Adaptive { shared: bool },
}

#[derive(Debug, Copy, Clone)]
pub struct Logger {
    dtlog: Real,
    te_0: Real,
    te_n: Real,
    instant: Instant,
    duration: Duration,
}
impl Logger {
    pub fn new(dtlog: Real) -> Logger {
        Logger {
            te_0: 0.0,
            te_n: 0.0,
            dtlog: dtlog,
            instant: Instant::now(),
            duration: Default::default(),
        }
    }
    fn setup(&mut self, psys: &ParticleSystem) {
        let ke = psys.kinectic_energy();
        let pe = psys.potential_energy();
        self.te_0 = ke + pe;
        self.te_n = ke + pe;
    }
    fn init(&mut self, tnow: Real, psys: &ParticleSystem) {
        println!(
            "#{:<11} #{:<10} #{:<10} #{:<10} #{:<10} #{:<10} \
             #{:<10} #{:<10} #{:<9} #{:<11}",
            "tnow", "ke", "pe", "ve", "err_0", "err_n", "rcom", "vcom", "elapsed", "elapsed_cum",
        );
        if tnow % self.dtlog == 0.0 {
            self.log(tnow, &psys);
        }
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

        let instant = Instant::now();
        let duration = instant.duration_since(self.instant);
        self.instant = instant;
        self.duration += duration;

        let elapsed = duration.subsec_nanos() as f64 * 1.0e-9 + duration.as_secs() as f64;
        let elapsed_cum =
            self.duration.subsec_nanos() as f64 * 1.0e-9 + self.duration.as_secs() as f64;

        println!(
            "{:<+12.6e} {:<+11.4e} {:<+11.4e} {:<+11.4e} {:<+11.4e} {:<+11.4e} \
             {:<+11.4e} {:<+11.4e} {:<+10.4e} {:<+12.6e}",
            tnow, ke, pe, ve, err_0, err_n, rcom, vcom, elapsed, elapsed_cum,
        );
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Simulation<I: Integrator + Debug> {
    tnow: Real,
    logger: Logger,
    integrator: I,
}
impl<I: Integrator + Debug> Simulation<I> {
    pub fn new(logger: Logger, integrator: I) -> Simulation<I> {
        Simulation {
            tnow: 0.0,
            logger: logger,
            integrator: integrator,
        }
    }
    pub fn init(&mut self, tnow: Real, psys: &mut ParticleSystem) {
        self.tnow = tnow;
        self.logger.setup(&psys);
        eprintln!("{:#?}", self);

        self.logger.init(tnow, &psys);
        self.integrator.init(tnow, &mut psys.particles[..]);
    }
    pub fn run(&mut self, tend: Real, psys: &mut ParticleSystem) {
        while self.tnow < tend {
            self.tnow += self.integrator.evolve(&mut psys.particles[..]);
            if self.tnow % self.logger.dtlog == 0.0 {
                self.logger.log(self.tnow, &psys);
            }
        }
        eprintln!("{:#?}", self);
    }
}

// -- end of file --
