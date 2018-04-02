use rand::Rng;
use real::{consts, Real};
use super::Sampler;

/// Plummer's stellar-density-profile
#[derive(Default)]
pub struct Plummer {}

impl Plummer {
    const MFRAC: Real = 0.999;
    const R_SCALE_FACTOR: Real = (3.0 * consts::PI) / 16.0;
    //    const V_SCALE_FACTOR: Real = (16.0 / (3.0 * consts::PI)).sqrt();
    const V_SCALE_FACTOR2: Real = 16.0 / (3.0 * consts::PI);

    pub fn new() -> Self {
        Default::default()
    }
}

impl Sampler for Plummer {
    type Output = (Real, Real);
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Output {
        let m_cum = rng.gen_range::<Real>(0.0, Self::MFRAC);
        let r = 1.0 / (m_cum.powf(-2.0 / 3.0) - 1.0).sqrt();

        let mut q: Real = 0.0;
        let mut w: Real = 0.1;
        while w > q * q * (1.0 - q * q).powf(3.5) {
            q = rng.gen_range::<Real>(0.0, 1.0);
            w = rng.gen_range::<Real>(0.0, 0.1);
        }
        let v = q * consts::SQRT_2 * (1.0 + r * r).powf(-0.25);
        (r * Self::R_SCALE_FACTOR, v * Self::V_SCALE_FACTOR2.sqrt())
    }
}

// -- end of file --
