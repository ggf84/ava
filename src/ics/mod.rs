pub mod imf;
pub mod sdp;

use crate::types::Real;
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

impl<IMF, SDP> Distribution<(Real, [Real; 3], [Real; 3])> for Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (Real, [Real; 3], [Real; 3]) {
        let m = self.imf.sample(rng);
        let (r, v) = self.sdp.sample(rng);
        (m, r, v)
    }
}

// -- end of file --
