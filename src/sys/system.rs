use compute;
use real::Real;
use sys::particles::Particle;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn len(&self) -> usize {
        self.particles.len()
    }
    pub fn sort_by_dt(&mut self, n: usize) {
        self.particles[..n].sort_unstable_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
    pub fn set_shared_dt(&mut self, dt: Real) {
        for p in self.particles.iter_mut() {
            p.dt = dt;
        }
    }
}

impl ParticleSystem {
    /// Compute the kinetic and potential energies of the system.
    pub fn energies(&self) -> (Real, Real) {
        let (ke, pe) = compute::energy::triangle(&self.particles[..]);
        let ke = 0.5 * ke.iter().sum::<Real>();
        let pe = 0.5 * pe.iter().sum::<Real>();
        let ke = 0.5 * ke / self.com_mass();
        (ke, pe)
    }
}

impl ParticleSystem {
    /// Compute the gravitational acceleration's (0)-derivative for each particle in the system due to itself.
    pub fn get_acc(&self) -> (Vec<[Real; 3]>,) {
        compute::acc::triangle(&self.particles[..])
    }
    /// Compute the gravitational acceleration's (0)-derivative for each particle in the system due to 'other' system.
    pub fn get_acc_p2p(&self, other: &Self) -> ((Vec<[Real; 3]>,), (Vec<[Real; 3]>,)) {
        compute::acc::rectangle(&self.particles[..], &other.particles[..])
    }

    /// Compute the gravitational acceleration's (0, 1)-derivatives for each particle in the system due to itself.
    pub fn get_jrk(&self) -> (Vec<[Real; 3]>, Vec<[Real; 3]>) {
        compute::jrk::triangle(&self.particles[..])
    }
    /// Compute the gravitational acceleration's (0, 1)-derivatives for each particle in the system due to 'other' system.
    pub fn get_jrk_p2p(
        &self,
        other: &Self,
    ) -> (
        (Vec<[Real; 3]>, Vec<[Real; 3]>),
        (Vec<[Real; 3]>, Vec<[Real; 3]>),
    ) {
        compute::jrk::rectangle(&self.particles[..], &other.particles[..])
    }

    /// Compute the gravitational acceleration's (0, 1, 2)-derivatives for each particle in the system due to itself.
    pub fn get_snp(&self) -> (Vec<[Real; 3]>, Vec<[Real; 3]>, Vec<[Real; 3]>) {
        compute::snp::triangle(&self.particles[..])
    }
    /// Compute the gravitational acceleration's (0, 1, 2)-derivatives for each particle in the system due to 'other' system.
    pub fn get_snp_p2p(
        &self,
        other: &Self,
    ) -> (
        (Vec<[Real; 3]>, Vec<[Real; 3]>, Vec<[Real; 3]>),
        (Vec<[Real; 3]>, Vec<[Real; 3]>, Vec<[Real; 3]>),
    ) {
        compute::snp::rectangle(&self.particles[..], &other.particles[..])
    }

    /// Compute the gravitational acceleration's (0, 1, 2, 3)-derivatives for each particle in the system due to itself.
    pub fn get_crk(
        &self,
    ) -> (
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
        Vec<[Real; 3]>,
    ) {
        compute::crk::triangle(&self.particles[..])
    }
    /// Compute the gravitational acceleration's (0, 1, 2, 3)-derivatives for each particle in the system due to 'other' system.
    pub fn get_crk_p2p(
        &self,
        other: &Self,
    ) -> (
        (
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
        ),
        (
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
            Vec<[Real; 3]>,
        ),
    ) {
        compute::crk::rectangle(&self.particles[..], &other.particles[..])
    }
}

/// Methods for center-of-mass determination and adjustment.
impl ParticleSystem {
    /// Get center-of-mass mass (a.k.a. total mass).
    pub fn com_mass(&self) -> Real {
        let mut msum = 0.0;
        for p in self.particles.iter() {
            msum += p.mass;
        }
        msum
    }
    /// Get center-of-mass position.
    pub fn com_pos(&self) -> [Real; 3] {
        let mut mtot = 0.0;
        let mut rcom = [0.0; 3];
        for p in self.particles.iter() {
            for k in 0..3 {
                rcom[k] += p.mass * p.pos[k];
            }
            mtot += p.mass;
        }
        rcom.iter_mut().for_each(|r| *r /= mtot);
        rcom
    }
    /// Get center-of-mass velocity.
    pub fn com_vel(&self) -> [Real; 3] {
        let mut mtot = 0.0;
        let mut vcom = [0.0; 3];
        for p in self.particles.iter() {
            for k in 0..3 {
                vcom[k] += p.mass * p.vel[k];
            }
            mtot += p.mass;
        }
        vcom.iter_mut().for_each(|v| *v /= mtot);
        vcom
    }
    /// Moves center-of-mass to the given coordinates.
    pub fn com_move_to(&mut self, pos: [Real; 3], vel: [Real; 3]) {
        for p in self.particles.iter_mut() {
            for k in 0..3 {
                p.pos[k] += pos[k];
                p.vel[k] += vel[k];
            }
        }
    }
    /// Moves center-of-mass to the origin of coordinates.
    pub fn com_to_origin(&mut self) {
        let mut rcom = self.com_pos();
        let mut vcom = self.com_vel();
        rcom.iter_mut().for_each(|r| *r = -*r);
        vcom.iter_mut().for_each(|v| *v = -*v);
        self.com_move_to(rcom, vcom);
    }
}

impl ParticleSystem {
    pub fn scale_mass(&mut self) {
        let mtot = self.com_mass();
        self.particles.iter_mut().for_each(|p| p.mass /= mtot);
    }

    pub fn scale_to_virial(&mut self, virial_ratio: Real) -> (Real, Real) {
        assert!(virial_ratio > 0.0);
        assert!(virial_ratio < 1.0);
        let (ke, pe) = self.energies();
        let v_scale = (-virial_ratio * pe / ke).sqrt();
        for p in self.particles.iter_mut() {
            for k in 0..3 {
                p.vel[k] *= v_scale;
            }
        }
        let ke = -virial_ratio * pe;
        (ke, pe)
    }

    pub fn scale_to_standard(&mut self, virial_ratio: Real) {
        let (ke, pe) = self.scale_to_virial(virial_ratio);
        let te = ke + pe;
        let r_scale = te / -0.25;
        let v_scale = 1.0 / r_scale.sqrt();
        for p in self.particles.iter_mut() {
            for k in 0..3 {
                p.pos[k] *= r_scale;
                p.vel[k] *= v_scale;
            }
        }
    }

    pub fn set_eps(&mut self, eps: Real) {
        // compute the original value of the virial ratio
        let (ke, pe) = self.energies();
        let virial_ratio = ke / -pe;

        // set the new eps
        let mmean = self.com_mass() / self.len() as Real;
        for p in self.particles.iter_mut() {
            p.eps = eps * (p.mass / mmean).cbrt();
        }

        // adjust the virial ratio to its original value
        self.scale_to_virial(virial_ratio);
    }
}

// -- end of file --
