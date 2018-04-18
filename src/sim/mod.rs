use real::Real;
use sys::particles::Particle;

pub mod hermite;

#[derive(Debug, Copy, Clone)]
pub enum TimeStepScheme {
    Constant { dt: Real },
    Adaptive { shared: bool },
}

pub trait Integrator {
    fn setup(&self, psys: &mut [Particle]);
    fn evolve(&self, psys: &mut [Particle]) -> Real;
}

// -- end of file --
