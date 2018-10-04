use crate::real::Real;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

/// Equal mass initial-mass-function
pub struct EqualMass {
    uniform: Uniform<Real>,
}

impl EqualMass {
    pub fn new(mass: Real) -> Self {
        assert!(mass > 0.0, "EqualMass::new called with `mass <= 0.0`");
        EqualMass {
            uniform: Uniform::new_inclusive(mass, mass),
        }
    }
}

impl Distribution<Real> for EqualMass {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Real {
        self.uniform.sample(rng)
    }
}

/// Maschberger (2013) initial-mass-function
pub struct Maschberger2013 {
    uniform: Uniform<Real>,
}

impl Maschberger2013 {
    const MU: Real = 0.2;
    const ALPHA: Real = 2.3;
    const BETA: Real = 1.4;

    pub fn new(mmin: Real, mmax: Real) -> Self {
        assert!(mmin > 0.0, "Maschberger2013::new called with `mmin <= 0.0`");
        assert!(mmax > 0.0, "Maschberger2013::new called with `mmax <= 0.0`");
        assert!(
            mmax >= mmin,
            "Maschberger2013::new called with `mmax < mmin`"
        );
        let gmin = (1.0 + (mmin / Self::MU).powf(1.0 - Self::ALPHA)).powf(1.0 - Self::BETA);
        let gmax = (1.0 + (mmax / Self::MU).powf(1.0 - Self::ALPHA)).powf(1.0 - Self::BETA);
        Maschberger2013 {
            uniform: Uniform::new_inclusive(gmin, gmax),
        }
    }
}

impl Distribution<Real> for Maschberger2013 {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Real {
        let g = self.uniform.sample(rng);
        Self::MU * (g.powf(1.0 / (1.0 - Self::BETA)) - 1.0).powf(1.0 / (1.0 - Self::ALPHA))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};

    fn use_imf<T, R: Rng>(imf: &T, rng: &mut R) -> Vec<Real>
    where
        T: Distribution<Real>,
    {
        imf.sample_iter(rng).map(|m| 2.0 * m).take(8).collect()
    }

    #[test]
    #[should_panic]
    fn equalmass_new() {
        let _ = EqualMass::new(-1.0);
    }

    #[test]
    #[should_panic]
    fn maschberger2013_new() {
        let _ = Maschberger2013::new(5.0, 1.0);
    }

    #[test]
    fn equalmass_sample() {
        let imf = EqualMass::new(1.0);

        let seed = [0; 32];
        let mut rng = StdRng::from_seed(seed);

        let m1: Vec<_> = imf.sample_iter(&mut rng).take(5).collect();
        let m2: Vec<_> = imf.sample_iter(&mut rng).take(3).collect();
        let mut rng = StdRng::from_seed(seed);
        let mm: Vec<_> = imf.sample_iter(&mut rng).take(8).collect();
        assert_eq!(m1.len() + m2.len(), mm.len());
        assert_eq!(&m1[..], &mm[..5]);
        assert_eq!(&m2[..], &mm[5..]);

        let r1 = use_imf(&imf, &mut rng);
        let mut rng = StdRng::from_seed(seed);
        let r2 = use_imf(&imf, &mut rng);
        assert_eq!(r1, r2);
        assert_eq!(r2, mm.iter().map(|m| 2.0 * m).collect::<Vec<_>>());

        let m: Vec<_> = imf.sample_iter(&mut rng).take(1_000_000).collect();
        let min = m.iter().fold(m[0], |p, q| p.min(*q));
        let max = m.iter().fold(m[0], |p, q| p.max(*q));
        assert!(min.to_bits() == max.to_bits());
    }

    #[test]
    fn maschberger2013_sample() {
        let imf = Maschberger2013::new(0.01, 150.0);

        let seed = [0; 32];
        let mut rng = StdRng::from_seed(seed);

        let m1: Vec<_> = imf.sample_iter(&mut rng).take(5).collect();
        let m2: Vec<_> = imf.sample_iter(&mut rng).take(3).collect();
        let mut rng = StdRng::from_seed(seed);
        let mm: Vec<_> = imf.sample_iter(&mut rng).take(8).collect();
        assert_eq!(m1.len() + m2.len(), mm.len());
        assert_eq!(&m1[..], &mm[..5]);
        assert_eq!(&m2[..], &mm[5..]);

        let r1 = use_imf(&imf, &mut rng);
        let mut rng = StdRng::from_seed(seed);
        let r2 = use_imf(&imf, &mut rng);
        assert_ne!(r1, r2);
        assert_eq!(r2, mm.iter().map(|m| 2.0 * m).collect::<Vec<_>>());

        let m: Vec<_> = imf.sample_iter(&mut rng).take(1_000_000).collect();
        let min = m.iter().fold(m[0], |p, q| p.min(*q));
        let max = m.iter().fold(m[0], |p, q| p.max(*q));
        assert!(min >= 0.01 && max <= 150.0);
    }
}

// -- end of file --
