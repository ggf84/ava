pub mod imf;
pub mod sdp;

use crate::{real::Real, sys::Particle};
use rand::{distributions::Distribution, Rng};

pub struct Model<IMF, SDP> {
    imf: IMF,
    sdp: SDP,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    pub fn new(imf: IMF, sdp: SDP) -> Self {
        Model { imf, sdp }
    }
}

impl<IMF, SDP> Distribution<Particle> for Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Particle {
        let m = self.imf.sample(rng);
        let (r, v) = self.sdp.sample(rng);
        Particle::new(m, r, v)
    }
}

// -- end of file --
