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
    eps_scale: Real,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Distribution<Real>,
    SDP: Distribution<([Real; 3], [Real; 3])>,
{
    pub fn new(imf: IMF, sdp: SDP, npart: usize, eps_scale: Real) -> Self {
        Model {
            imf: imf,
            sdp: sdp,
            npart: npart,
            eps_scale: eps_scale,
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> ParticleSystem {
        let mut psys = ParticleSystem::new();
        psys.particles = self.sample_iter(rng).take(self.npart).collect();

        let (mtot, [rx, ry, rz], [vx, vy, vz]) = psys.com_mass_pos_vel();
        // let mtot = psys.scale_mass(1.0 / self.npart as Real);
        let mtot = psys.scale_mass(1.0 / mtot);
        psys.com_move_by([-rx, -ry, -rz], [-vx, -vy, -vz]); // reset center-of-mass to the origin of coordinates.

        let q_vir = 0.5;
        let (ke, pe) = psys.energies();
        let rvir = psys.scale_to_standard(q_vir, ke, pe, mtot);

        if self.eps_scale > 0.0 {
            psys.set_eps(self.eps_scale, rvir);
            let (ke, pe) = psys.energies();
            let _ = psys.scale_to_standard(q_vir, ke, pe, mtot);
        }

        let (ke, pe) = psys.energies();
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
