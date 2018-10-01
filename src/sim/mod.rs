mod hermite;

use self::hermite::{Hermite4, Hermite6, Hermite8};
use crate::{gravity::Energy, real::Real, sys::ParticleSystem};
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

trait Evolver {
    fn init(&self, dtmax: Real, psys: &mut ParticleSystem);
    fn evolve(&self, dtmax: Real, psys: &mut ParticleSystem) -> Counter;
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct Counter {
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

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum TimeStepScheme {
    Constant { dt: Real },
    Adaptive { shared: bool },
}
impl TimeStepScheme {
    pub fn constant(dt: Real) -> Self {
        TimeStepScheme::Constant {
            dt: to_power_of_two(dt),
        }
    }
    pub fn adaptive_shared() -> Self {
        TimeStepScheme::Adaptive { shared: true }
    }
    pub fn adaptive_block() -> Self {
        TimeStepScheme::Adaptive { shared: false }
    }
    fn match_dt(&self, psys: &mut ParticleSystem) -> Real {
        match *self {
            TimeStepScheme::Constant { dt } => {
                psys.attrs.dt.iter_mut().for_each(|idt| *idt = dt);
                dt
            }
            TimeStepScheme::Adaptive { shared } => {
                let dt = psys.attrs.dt[0];
                if shared {
                    psys.attrs.dt.iter_mut().for_each(|idt| *idt = dt);
                }
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
    pub fn hermite4(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Integrator::H4(Hermite4::new(tstep_scheme, eta, npec))
    }
    pub fn hermite6(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Integrator::H6(Hermite6::new(tstep_scheme, eta, npec))
    }
    pub fn hermite8(tstep_scheme: TimeStepScheme, eta: Real, npec: u8) -> Self {
        Integrator::H8(Hermite8::new(tstep_scheme, eta, npec))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Simulation {
    psys: ParticleSystem,
    integrator: Integrator,
    dtres: Real,
    dtlog: Real,
    dtmax: Real,
    logger: Logger,
}
impl Simulation {
    pub fn new(
        mut psys: ParticleSystem,
        integrator: Integrator,
        dtres_pow: i32,
        dtlog_pow: i32,
        dtmax_pow: i32,
        instant: &mut Instant,
    ) -> Self {
        assert!(dtres_pow >= dtlog_pow);
        assert!(dtlog_pow >= dtmax_pow);
        let dtres = (2.0 as Real).powi(dtres_pow);
        let dtlog = (2.0 as Real).powi(dtlog_pow);
        let dtmax = (2.0 as Real).powi(dtmax_pow);

        match &integrator {
            Integrator::H4(integrator) => integrator.init(dtmax, &mut psys),
            Integrator::H6(integrator) => integrator.init(dtmax, &mut psys),
            Integrator::H8(integrator) => integrator.init(dtmax, &mut psys),
        }

        let mut logger = Logger::new(&psys);
        logger.print_log(&psys, instant);

        Simulation {
            psys,
            integrator,
            dtres,
            dtlog,
            dtmax,
            logger,
        }
    }
    pub fn evolve(&mut self, tend: Real, instant: &mut Instant) -> Result<(), std::io::Error> {
        while self.psys.time < tend {
            let counter = match &self.integrator {
                Integrator::H4(integrator) => integrator.evolve(self.dtmax, &mut self.psys),
                Integrator::H6(integrator) => integrator.evolve(self.dtmax, &mut self.psys),
                Integrator::H8(integrator) => integrator.evolve(self.dtmax, &mut self.psys),
            };
            self.logger.counter += counter;

            if self.psys.time % self.dtlog == 0.0 {
                self.logger.print_log(&self.psys, instant);
            }
            if self.psys.time % self.dtres == 0.0 {
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
        let (ke, pe) = Energy::new(mtot).energies(psys.as_ref());
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
            counter: Default::default(),
            counter_cum: Default::default(),
            duration: Default::default(),
            duration_cum: Default::default(),
        }
    }
    fn print_log(&mut self, psys: &ParticleSystem, instant: &mut Instant) {
        let (mtot, rcom, vcom) = psys.com_mass_pos_vel();
        let rcom = rcom.iter().fold(0.0, |s, r| s + r * r).sqrt();
        let vcom = vcom.iter().fold(0.0, |s, v| s + v * v).sqrt();
        let (ke, pe) = Energy::new(mtot).energies(psys.as_ref());
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
        self.counter = Counter::new();
    }
}

// -- end of file --
