use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use real::{consts, Real};

/// Plummer's stellar-density-profile
pub struct Plummer {
    m_uniform: Uniform<Real>,
    q_uniform: Uniform<Real>,
    w_uniform: Uniform<Real>,
}

impl Plummer {
    const MFRAC: Real = 0.999;
    const R_SCALE_FACTOR: Real = (3.0 * consts::PI) / 16.0;
    //    const V_SCALE_FACTOR: Real = (16.0 / (3.0 * consts::PI)).sqrt();
    const V_SCALE_FACTOR2: Real = 16.0 / (3.0 * consts::PI);

    pub fn new() -> Self {
        Plummer {
            m_uniform: Uniform::new_inclusive(0.0, Self::MFRAC),
            q_uniform: Uniform::new_inclusive(0.0, 1.0),
            w_uniform: Uniform::new_inclusive(0.0, 0.1),
        }
    }
}

impl Distribution<([Real; 3], [Real; 3])> for Plummer {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ([Real; 3], [Real; 3]) {
        let m = self.m_uniform.sample(rng);
        let r = 1.0 / (m.powf(-2.0 / 3.0) - 1.0).sqrt();

        let mut q: Real = 0.0;
        let mut w: Real = 1.0;
        while w > q * q * (1.0 - q * q).powf(3.5) {
            q = self.q_uniform.sample(rng);
            w = self.w_uniform.sample(rng);
        }
        let v = q * consts::SQRT_2 * (1.0 + r * r).powf(-0.25);

        let r = r * Self::R_SCALE_FACTOR;
        let v = v * Self::V_SCALE_FACTOR2.sqrt();
        (to_xyz(r, rng), to_xyz(v, rng))
    }
}

fn to_xyz<R: Rng + ?Sized>(val: Real, rng: &mut R) -> [Real; 3] {
    let theta = rng.gen_range::<Real>(-1.0, 1.0).acos();
    let phi = rng.gen_range::<Real>(0.0, 2.0 * consts::PI);
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    [val * st * cp, val * st * sp, val * ct]
}

// -- end of file --
