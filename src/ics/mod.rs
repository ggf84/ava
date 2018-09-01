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
    eps_factor: Real,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    pub fn new(imf: IMF, sdp: SDP, npart: usize, eps_factor: Real) -> Self {
        Model {
            imf: imf,
            sdp: sdp,
            npart: npart,
            eps_factor: eps_factor,
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> ParticleSystem {
        let mut psys = ParticleSystem::new();
        psys.particles = self.sample_iter(rng).take(self.npart).collect();

        // psys.scale_mass(1.0 / self.npart as Real);

        let mtot = psys.com_mass();
        psys.scale_mass(1.0 / mtot);

        let q_vir = 0.5;
        let mtot = psys.com_mass();
        psys.com_to_origin();

        let (ke, pe) = psys.energies();
        psys.scale_to_virial(ke, pe, q_vir);
        psys.scale_to_standard(mtot, pe, q_vir);

        if self.eps_factor > 0.0 {
            psys.set_eps(self.eps_factor);
            let (ke, pe) = psys.energies();
            psys.scale_to_virial(ke, pe, q_vir);
            psys.scale_to_standard(mtot, pe, q_vir);
        }

        let (ke, pe) = psys.energies();
        let rvir = mtot.powi(2) / (-2.0 * pe);
        eprintln!("{:?} {:?} {:?} {:?} {:?}", mtot, ke, pe, ke + pe, rvir);

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
