use rand::{distributions::Distribution, Rng};
use real::Real;
use sys::particles::Particle;
use sys::system::ParticleSystem;

pub mod imf;
pub mod sdp;

pub struct Model<IMF, SDP> {
    imf: IMF,
    sdp: SDP,
    npart: usize,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    pub fn new(imf: IMF, sdp: SDP, npart: usize) -> Self {
        Model {
            imf: imf,
            sdp: sdp,
            npart: npart,
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> ParticleSystem {
        let mut psys = ParticleSystem::new();
        psys.particles = self.sample_iter(rng).take(self.npart).collect();

        psys.scale_mass();
        psys.com_to_origin();
        psys.scale_to_standard(0.5);
        psys
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
