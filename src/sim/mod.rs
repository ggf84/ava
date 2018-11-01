mod hermite;

use self::hermite::{Hermite4, Hermite6, Hermite8};
use crate::{
    gravity::{
        energy::{Energy, EnergyKernel},
        Compute,
    },
    real::Real,
    sys::ParticleSystem,
};
use serde_derive::{Deserialize, Serialize};
use std::{
    fs::File,
    io::BufWriter,
    ops::AddAssign,
    time::{Duration, Instant},
};

fn to_power_of_two(dt: Real) -> Real {
    let pow = dt.log2().floor();
    (2.0 as Real).powi(pow as i32)
}

trait Evolver {
    fn init(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem);
    fn evolve(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem) -> Counter;
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct Counter {
    steps: u16,
    bsteps: u32,
    isteps: u64,
}
impl AddAssign for Counter {
    fn add_assign(&mut self, other: Counter) {
        self.steps += other.steps;
        self.bsteps += other.bsteps;
        self.isteps += other.isteps;
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Scheme {
    Constant,
    Variable,
    Individual,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeStepScheme {
    eta: Real,
    dtres: Real,
    dtlog: Real,
    dtmax: Real,
    scheme: Scheme,
}
impl TimeStepScheme {
    pub fn constant(eta: Real, dtres_pow: i32, dtlog_pow: i32, dtmax_pow: i32) -> Self {
        Self::new(eta, dtres_pow, dtlog_pow, dtmax_pow, Scheme::Constant)
    }

    pub fn variable(eta: Real, dtres_pow: i32, dtlog_pow: i32, dtmax_pow: i32) -> Self {
        Self::new(eta, dtres_pow, dtlog_pow, dtmax_pow, Scheme::Variable)
    }

    pub fn individual(eta: Real, dtres_pow: i32, dtlog_pow: i32, dtmax_pow: i32) -> Self {
        Self::new(eta, dtres_pow, dtlog_pow, dtmax_pow, Scheme::Individual)
    }

    fn new(eta: Real, dtres_pow: i32, dtlog_pow: i32, dtmax_pow: i32, scheme: Scheme) -> Self {
        assert!(dtres_pow >= dtlog_pow);
        assert!(dtlog_pow >= dtmax_pow);
        TimeStepScheme {
            eta,
            dtres: (2.0 as Real).powi(dtres_pow),
            dtlog: (2.0 as Real).powi(dtlog_pow),
            dtmax: (2.0 as Real).powi(dtmax_pow),
            scheme,
        }
    }

    fn match_dt(&self, psys: &mut ParticleSystem) -> Real {
        match self.scheme {
            Scheme::Constant => {
                let dt = to_power_of_two(self.eta * self.dtmax);
                psys.attrs.dt.iter_mut().for_each(|idt| *idt = dt);
                dt
            }
            Scheme::Variable => {
                let dt = psys.attrs.dt[0];
                psys.attrs.dt.iter_mut().for_each(|idt| *idt = dt);
                dt
            }
            Scheme::Individual => {
                let dt = psys.attrs.dt[0];
                dt
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum Integrator {
    H4(Hermite4),
    H6(Hermite6),
    H8(Hermite8),
}
impl Integrator {
    pub fn hermite4(npec: u8) -> Self {
        Integrator::H4(Hermite4::new(npec))
    }

    pub fn hermite6(npec: u8) -> Self {
        Integrator::H6(Hermite6::new(npec))
    }

    pub fn hermite8(npec: u8) -> Self {
        Integrator::H8(Hermite8::new(npec))
    }
}
impl Evolver for Integrator {
    fn init(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem) {
        match self {
            Integrator::H4(integrator) => integrator.init(tstep_scheme, psys),
            Integrator::H6(integrator) => integrator.init(tstep_scheme, psys),
            Integrator::H8(integrator) => integrator.init(tstep_scheme, psys),
        }
    }

    fn evolve(&self, tstep_scheme: &TimeStepScheme, psys: &mut ParticleSystem) -> Counter {
        match self {
            Integrator::H4(integrator) => integrator.evolve(tstep_scheme, psys),
            Integrator::H6(integrator) => integrator.evolve(tstep_scheme, psys),
            Integrator::H8(integrator) => integrator.evolve(tstep_scheme, psys),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Simulation {
    psys: ParticleSystem,
    integrator: Integrator,
    tstep_scheme: TimeStepScheme,
    logger: Logger,
}
impl Simulation {
    pub fn new(
        mut psys: ParticleSystem,
        integrator: Integrator,
        tstep_scheme: TimeStepScheme,
        instant: &mut Instant,
    ) -> Self {
        integrator.init(&tstep_scheme, &mut psys);

        let mut logger = Logger::new(&psys);
        logger.print_log(&psys, instant);

        Simulation {
            psys,
            integrator,
            tstep_scheme,
            logger,
        }
    }

    pub fn evolve(&mut self, tend: Real, instant: &mut Instant) -> Result<(), std::io::Error> {
        while self.psys.time < tend {
            self.logger.counter += self.integrator.evolve(&self.tstep_scheme, &mut self.psys);

            if self.psys.time % self.tstep_scheme.dtlog == 0.0 {
                self.logger.print_log(&self.psys, instant);
            }
            if self.psys.time % self.tstep_scheme.dtres == 0.0 {
                self.write_restart_file()?;
            }
        }
        Ok(())
    }

    fn write_restart_file(&self) -> Result<(), std::io::Error> {
        let file = File::create("res.sim")?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &self).unwrap();
        Ok(())
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Logger {
    te_0: Real,
    te_n: Real,
    counter: Counter,
    counter_cum: Counter,
    duration: Duration,
    duration_cum: Duration,
}
impl Logger {
    fn new(psys: &ParticleSystem) -> Self {
        let mtot = psys.com_mass();
        let mut energy = Energy::zeros(psys.len());
        EnergyKernel {}.compute(&psys.as_slice(), &mut energy.as_mut_slice());
        let (ke, pe) = energy.reduce(mtot);
        let te = ke + pe;
        let te_0 = te;
        let te_n = te;

        let names = [
            "        time",
            "          ke",
            "          pe",
            "          ve",
            "     err_rel",
            "     err_cum",
            "        rcom",
            "        vcom",
            "      bsteps",
            "      isteps",
            "  bsteps_cum",
            "  isteps_cum",
            "    duration",
            "duration_cum",
        ];
        let mut h0 = String::from("");
        let mut h1 = String::from("");
        for (i, name) in names.iter().enumerate() {
            let width = name.len();
            let s0 = format!(" {:>width$}", format!("({})", i + 1), width = width);
            let s1 = format!(" {:>width$}", name, width = width);
            h0.push_str(&s0);
            h1.push_str(&s1);
        }
        println!("#{}\n#{}\n# {}", h0, h1, "=".repeat(h1.len() - 1));

        Logger {
            te_0,
            te_n,
            counter: Counter::default(),
            counter_cum: Counter::default(),
            duration: Duration::default(),
            duration_cum: Duration::default(),
        }
    }

    fn print_log(&mut self, psys: &ParticleSystem, instant: &mut Instant) {
        let (mtot, rcom, vcom) = psys.com_mass_pos_vel();
        let rcom = rcom.iter().fold(0.0, |s, r| s + r * r).sqrt();
        let vcom = vcom.iter().fold(0.0, |s, v| s + v * v).sqrt();
        let mut energy = Energy::zeros(psys.len());
        EnergyKernel {}.compute(&psys.as_slice(), &mut energy.as_mut_slice());
        let (ke, pe) = energy.reduce(mtot);
        let te = ke + pe;
        let ve = 2.0 * ke + pe;
        let err_rel = (te - self.te_n) / self.te_n;
        let err_cum = (te - self.te_0) / self.te_0;
        self.te_n = te;

        let now = Instant::now();
        self.duration = now.duration_since(*instant);
        *instant = now;

        self.counter_cum += self.counter;
        self.duration_cum += self.duration;

        let duration = (1_000_000_000 * u128::from(self.duration.as_secs())
            + u128::from(self.duration.subsec_nanos())) as f64
            * 1.0e-9;
        let duration_cum = (1_000_000_000 * u128::from(self.duration_cum.as_secs())
            + u128::from(self.duration_cum.subsec_nanos())) as f64
            * 1.0e-9;

        let line = format!(
            "{:>width$.prec$} {:>width$.prec$e} {:>width$.prec$e} {:>width$.prec$e} \
             {:>width$.prec$e} {:>width$.prec$e} {:>width$.prec$e} {:>width$.prec$e} \
             {:>width$} {:>width$} {:>width$} {:>width$} {:>width$.prec$} {:>width$.prec$}",
            psys.time,
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
            width = 12,
            prec = 4,
        );
        println!("  {}", line);
        self.counter = Counter::default();
    }
}

// -- end of file --
