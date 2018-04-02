use rand::Rng;
use real::Real;
use super::Sampler;

/// Equal mass initial-mass-function
pub struct EqualMass {
    mass: Real,
}

impl EqualMass {
    pub fn new(mass: Real) -> Self {
        assert!(mass > 0.0, "EqualMass::new called with `mass <= 0.0`");
        EqualMass { mass: mass }
    }
}

impl Sampler for EqualMass {
    type Output = Real;
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Output {
        // It'd be simpler to just return self.mass. But we do this way in order
        // to guarantee that the rng state would be the same as if we were using
        // another IMF sampler.
        self.mass + 0.0 * rng.gen::<Real>()
    }
}

/// Maschberger (2013) initial-mass-function
pub struct Maschberger2013 {
    gmin: Real,
    gmax: Real,
}

impl Maschberger2013 {
    const MU: Real = 0.2;
    const ALPHA: Real = 2.3;
    const BETA: Real = 1.4;

    pub fn new(mmin: Real, mmax: Real) -> Self {
        assert!(mmin > 0.0, "Maschberger2013::new called with `mmin <= 0.0`");
        assert!(
            mmin < mmax,
            "Maschberger2013::new called with `mmin >= mmax`"
        );
        Maschberger2013 {
            gmin: (1.0 + (mmin / Self::MU).powf(1.0 - Self::ALPHA)).powf(1.0 - Self::BETA),
            gmax: (1.0 + (mmax / Self::MU).powf(1.0 - Self::ALPHA)).powf(1.0 - Self::BETA),
        }
    }
}

impl Sampler for Maschberger2013 {
    type Output = Real;
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Output {
        let g = rng.gen_range::<Real>(self.gmin, self.gmax);
        Self::MU * (g.powf(1.0 / (1.0 - Self::BETA)) - 1.0).powf(1.0 / (1.0 - Self::ALPHA))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};

    fn use_imf<T, R: Rng>(imf: &T, rng: &mut R) -> Vec<Real>
    where
        T: Sampler<Output = Real>,
    {
        (0..8)
            .map(|_| imf.sample(rng))
            .map(|m| 2.0 * m)
            .collect::<Vec<_>>()
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
        let seed = [1, 2, 3, 4];
        let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

        let imf = EqualMass::new(1.0);

        let m1 = (0..5).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        let m2 = (0..3).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        rng.reseed(&seed[..]);
        let mm = (0..8).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        assert_eq!(m1.len() + m2.len(), mm.len());
        assert_eq!(&m1[..], &mm[..5]);
        assert_eq!(&m2[..], &mm[5..]);

        let r1 = use_imf(&imf, &mut rng);
        rng.reseed(&seed[..]);
        let r2 = use_imf(&imf, &mut rng);
        assert_eq!(r1, r2);
        assert_eq!(r2, mm.iter().map(|m| 2.0 * m).collect::<Vec<_>>());

        let m = (0..1_000_000)
            .map(|_| imf.sample(&mut rng))
            .collect::<Vec<_>>();
        let min = m.iter().fold(m[0], |p, q| p.min(*q));
        let max = m.iter().fold(m[0], |p, q| p.max(*q));
        assert!(min == max);
    }

    #[test]
    fn maschberger2013_sample() {
        let seed = [1, 2, 3, 4];
        let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

        let imf = Maschberger2013::new(0.01, 150.0);

        let m1 = (0..5).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        let m2 = (0..3).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        rng.reseed(&seed[..]);
        let mm = (0..8).map(|_| imf.sample(&mut rng)).collect::<Vec<_>>();
        assert_eq!(m1.len() + m2.len(), mm.len());
        assert_eq!(&m1[..], &mm[..5]);
        assert_eq!(&m2[..], &mm[5..]);

        let r1 = use_imf(&imf, &mut rng);
        rng.reseed(&seed[..]);
        let r2 = use_imf(&imf, &mut rng);
        assert_ne!(r1, r2);
        assert_eq!(r2, mm.iter().map(|m| 2.0 * m).collect::<Vec<_>>());

        let m = (0..1_000_000)
            .map(|_| imf.sample(&mut rng))
            .collect::<Vec<_>>();
        let min = m.iter().fold(m[0], |p, q| p.min(*q));
        let max = m.iter().fold(m[0], |p, q| p.max(*q));
        assert!(min >= 0.01 && max < 150.0);
    }
}

// -- end of file --
