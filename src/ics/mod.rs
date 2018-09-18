pub mod imf;
pub mod sdp;

use crate::{
    gravity::Energy,
    real::Real,
    sys::{Particle, ParticleSystem},
};
use rand::{distributions::Distribution, Rng};

pub struct Model<IMF, SDP> {
    npart: usize,
    imf: IMF,
    sdp: SDP,
    q_vir: Real,
    eps_param: Option<Real>,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    pub fn new(npart: usize, imf: IMF, sdp: SDP, q_vir: Real, eps_param: Option<Real>) -> Self {
        Model {
            npart,
            imf,
            sdp,
            q_vir,
            eps_param,
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> ParticleSystem {
        let mut psys = ParticleSystem::new();
        psys.particles = self.sample_iter(rng).take(self.npart).collect();

        let (mtot, [rx, ry, rz], [vx, vy, vz]) = psys.com_mass_pos_vel();
        let mtot = psys.scale_mass(1.0 / mtot);
        psys.com_move_by([-rx, -ry, -rz], [-vx, -vy, -vz]); // reset center-of-mass to the origin of coordinates.

        if let Some(eps_param) = self.eps_param {
            psys.set_eps(eps_param * 1.0); // assume rvir == 1
        }
        let (ke, pe) = Energy::new(mtot).energies(psys.as_slice());
        let ke = psys.scale_to_virial(self.q_vir, ke, pe);
        psys.scale_to_standard(-0.25, ke + pe);

        let (ke, pe) = Energy::new(mtot).energies(psys.as_slice());
        let rvir = mtot.powi(2) / (-2.0 * pe);
        eprintln!(
            "mtot: {:?}\nke: {:?}\npe: {:?}\nte: {:?}\nrvir: {:?}",
            mtot,
            ke,
            pe,
            ke + pe,
            rvir
        );

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
