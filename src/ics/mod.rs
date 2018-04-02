pub mod imf;

use rand::Rng;

pub trait Sampler {
    type Output;
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Output;
}

// -- end of file --
