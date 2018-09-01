use rand::{
    distributions::{Distribution, Normal, Uniform},
    Rng,
};
use real::{consts, Real};

/// Plummer's stellar-density-profile
pub struct Plummer {
    m_uniform: Uniform<Real>,
    // q_uniform: Uniform<Real>,
    // w_uniform: Uniform<Real>,
}

impl Plummer {
    const R_SCALE_FACTOR: Real = (3.0 * consts::PI) / 16.0;
    const V2_SCALE_FACTOR: Real = 16.0 / (3.0 * consts::PI);
    // const WMAX: Real = (686.0 / 19683.0) * (7.0 as Real).sqrt();
    // const WMAX2: Real = 3294172.0 / 387420489.0;

    pub fn new() -> Self {
        Plummer {
            m_uniform: Uniform::new(0.0, 1.0),
            // q_uniform: Uniform::new(0.0, 1.0),
            // w_uniform: Uniform::new(0.0, Self::WMAX2.sqrt()),
        }
    }
}

impl Distribution<([Real; 3], [Real; 3])> for Plummer {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ([Real; 3], [Real; 3]) {
        let m = self.m_uniform.sample(rng);
        let r = 1.0 / (m.powf(-2.0 / 3.0) - 1.0).sqrt();
        let [rx, ry, rz] = to_xyz(r * Self::R_SCALE_FACTOR, rng);

        // let mut q: Real = 0.0;
        // let mut w: Real = 1.0;
        // while w >= q * q * (1.0 - q * q).powf(3.5) {
        //     q = self.q_uniform.sample(rng);
        //     w = self.w_uniform.sample(rng);
        // }
        // let v2 = q * q * 2.0 / (1.0 + r * r).sqrt();
        // let [vx, vy, vz] = to_xyz((v2 * Self::V2_SCALE_FACTOR).sqrt(), rng);

        let sigma2_1d = 1.0 / (6.0 * (1.0 + r * r).sqrt());
        let v_normal = Normal::new(0.0, (sigma2_1d * Self::V2_SCALE_FACTOR).sqrt());
        let vx = v_normal.sample(rng);
        let vy = v_normal.sample(rng);
        let vz = v_normal.sample(rng);

        ([rx, ry, rz], [vx, vy, vz])
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
