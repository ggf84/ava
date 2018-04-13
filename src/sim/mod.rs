use real::Real;
use sys::system::ParticleSystem;

pub mod hermite;

pub enum TimeStepScheme {
    Constant,
    Adaptive { shared: bool },
}

pub trait Integrator {
    fn setup(&self, psys: &mut ParticleSystem);
    fn evolve(&self, psys: &mut ParticleSystem) -> Real;
}

// -- end of file --
