use rand::{
    distributions::{Distribution, Normal, Uniform},
    Rng,
};
use real::{consts, Real};

trait SDP {
    const R_SCALE_FACTOR: Real;
    const V2_SCALE_FACTOR: Real;
    /// Mass as a function of the radius.
    fn mass_r(r: Real) -> Real;
    /// Radius as a function of the mass.
    fn radius_m(m: Real) -> Real;
    /// 1-D (isotropic) velocity dispersion as a function of the radius.
    fn sigma2_1d(r: Real) -> Real;
}

/// Plummer's density profile.
pub struct Plummer {
    m_uniform: Uniform<Real>,
}
impl SDP for Plummer {
    const R_SCALE_FACTOR: Real = (3.0 * consts::PI) / 16.0;
    const V2_SCALE_FACTOR: Real = 16.0 / (3.0 * consts::PI);
    fn mass_r(r: Real) -> Real {
        (r / (r * r + 1.0).sqrt()).powi(3)
    }
    fn radius_m(m: Real) -> Real {
        1.0 / (m.powf(-2.0 / 3.0) - 1.0).sqrt()
    }
    fn sigma2_1d(r: Real) -> Real {
        1.0 / (6.0 * (r * r + 1.0).sqrt())
    }
}

/// Dehnen's (gamma = 0) density profile.
pub struct Dehnen0 {
    m_uniform: Uniform<Real>,
}
impl SDP for Dehnen0 {
    const R_SCALE_FACTOR: Real = 1.0 / 5.0;
    const V2_SCALE_FACTOR: Real = 5.0;
    fn mass_r(r: Real) -> Real {
        (r / (r + 1.0)).powi(3)
    }
    fn radius_m(m: Real) -> Real {
        let m_13 = m.cbrt();
        m_13 / (1.0 - m_13)
    }
    fn sigma2_1d(r: Real) -> Real {
        ((1.0 + 6.0 * r) / (r + 1.0).powi(2)) / 30.0
    }
}

/// Dehnen's (gamma = 1/2) density profile.
pub struct Dehnen12 {
    m_uniform: Uniform<Real>,
}
impl SDP for Dehnen12 {
    const R_SCALE_FACTOR: Real = 1.0 / 4.0;
    const V2_SCALE_FACTOR: Real = 4.0;
    fn mass_r(r: Real) -> Real {
        (r / (r + 1.0)).powf(5.0 / 2.0)
    }
    fn radius_m(m: Real) -> Real {
        let m_25 = m.powf(2.0 / 5.0);
        m_25 / (1.0 - m_25)
    }
    fn sigma2_1d(r: Real) -> Real {
        (r / (r + 1.0).powi(3)).sqrt() / 5.0
    }
}

/// Dehnen's (gamma = 1) density profile (a.k.a. Hernquist model).
pub struct Dehnen1 {
    m_uniform: Uniform<Real>,
}
impl SDP for Dehnen1 {
    const R_SCALE_FACTOR: Real = 1.0 / 3.0;
    const V2_SCALE_FACTOR: Real = 3.0;
    fn mass_r(r: Real) -> Real {
        (r / (r + 1.0)).powi(2)
    }
    fn radius_m(m: Real) -> Real {
        let m_12 = m.sqrt();
        m_12 / (1.0 - m_12)
    }
    fn sigma2_1d(r: Real) -> Real {
        ((r + 1.0).powi(4) * (1.0 / r).ln_1p()
            - (((r + 7.0 / 2.0) * r + 13.0 / 3.0) * r + 25.0 / 12.0)) * r / (r + 1.0)
    }
}

/// Dehnen's (gamma = 3/2) density profile.
pub struct Dehnen32 {
    m_uniform: Uniform<Real>,
}
impl SDP for Dehnen32 {
    const R_SCALE_FACTOR: Real = 1.0 / 2.0;
    const V2_SCALE_FACTOR: Real = 2.0;
    fn mass_r(r: Real) -> Real {
        (r / (r + 1.0)).powf(3.0 / 2.0)
    }
    fn radius_m(m: Real) -> Real {
        let m_23 = m.powf(2.0 / 3.0);
        m_23 / (1.0 - m_23)
    }
    fn sigma2_1d(r: Real) -> Real {
        (-4.0 * (r * (r + 1.0).powi(3)) * (1.0 / r).ln_1p()
            + (((4.0 * r + 10.0) * r + 22.0 / 3.0) * r + 1.0)) * (r / (r + 1.0)).sqrt()
    }
}

/// Dehnen's (gamma = 2) density profile (a.k.a. Jaffe model).
pub struct Dehnen2 {
    m_uniform: Uniform<Real>,
}
impl SDP for Dehnen2 {
    const R_SCALE_FACTOR: Real = 1.0;
    const V2_SCALE_FACTOR: Real = 1.0;
    fn mass_r(r: Real) -> Real {
        r / (r + 1.0)
    }
    fn radius_m(m: Real) -> Real {
        m / (1.0 - m)
    }
    fn sigma2_1d(r: Real) -> Real {
        6.0 * (r * (r + 1.0)).powi(2) * (1.0 / r).ln_1p()
            - (((6.0 * r + 9.0) * r + 2.0) * r - 1.0 / 2.0)
    }
}

/// Hernquist's density profile.
pub type Hernquist = Dehnen1;

/// Jaffe's density profile.
pub type Jaffe = Dehnen2;

macro_rules! impl_distribution {
    ($($name: ident),*) => {
        $(
            impl $name {
                pub fn new() -> Self {
                    let rh = Self::radius_m(0.5);
                    let mfrac = Self::mass_r(50.0 * rh);
                    $name {
                        m_uniform: Uniform::new(0.0, mfrac),
                    }
                }
            }
            impl Distribution<([Real; 3], [Real; 3])> for $name {
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ([Real; 3], [Real; 3]) {
                    let m = self.m_uniform.sample(rng);
                    let r = Self::radius_m(m);
                    let [rx, ry, rz] = to_xyz(r * Self::R_SCALE_FACTOR, rng);

                    let sigma2_1d = Self::sigma2_1d(r);
                    let v_normal = Normal::new(0.0, (sigma2_1d * Self::V2_SCALE_FACTOR).sqrt());
                    let vx = v_normal.sample(rng);
                    let vy = v_normal.sample(rng);
                    let vz = v_normal.sample(rng);

                    ([rx, ry, rz], [vx, vy, vz])
                }
            }
        )*
    }
}
impl_distribution!(Plummer, Dehnen0, Dehnen12, Dehnen1, Dehnen32, Dehnen2);

fn to_xyz<R: Rng + ?Sized>(val: Real, rng: &mut R) -> [Real; 3] {
    let theta = rng.gen_range::<Real>(-1.0, 1.0).acos();
    let phi = rng.gen_range::<Real>(0.0, 2.0 * consts::PI);
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    [val * st * cp, val * st * sp, val * ct]
}

// -- end of file --
