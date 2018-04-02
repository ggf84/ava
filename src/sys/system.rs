use real::Real;
use compute;
use sys::particles::Particle;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Default::default()
    }
}

impl ParticleSystem {
    /// Compute the gravitational potential for each particle in the system due to itself.
    pub fn get_phi(&self) -> Vec<Real> {
        compute::phi::triangle(&self.particles[..])
    }
    /// Compute the gravitational potential for each particle in the system due to 'other' system.
    pub fn get_phi_p2p(&self, other: &Self) -> (Vec<Real>, Vec<Real>) {
        compute::phi::rectangle(&self.particles[..], &other.particles[..])
    }

    /// Compute the [0]-derivative of the gravitational acceleration for each particle in the system due to itself.
    pub fn get_acc(&self) -> Vec<([Real; 3],)> {
        compute::acc::triangle(&self.particles[..])
    }
    /// Compute the [0]-derivative of the gravitational acceleration for each particle in the system due to 'other' system.
    pub fn get_acc_p2p(&self, other: &Self) -> (Vec<([Real; 3],)>, Vec<([Real; 3],)>) {
        compute::acc::rectangle(&self.particles[..], &other.particles[..])
    }
}

impl ParticleSystem {
    /// Compute the kinetic energy of the system
    pub fn kinectic_energy(&self) -> Real {
        let mut ke = 0.0;
        for p in self.particles.iter() {
            let mut v2 = 0.0;
            v2 += p.r.1[0] * p.r.1[0];
            v2 += p.r.1[1] * p.r.1[1];
            v2 += p.r.1[2] * p.r.1[2];
            ke += p.m * v2;
        }
        0.5 * ke
    }
    /// Compute the potential energy of the system
    pub fn potential_energy(&self) -> Real {
        let phi = self.get_phi();
        let pe = self.particles
            .iter()
            .zip(phi.iter())
            .map(|(p, phi)| p.m * phi)
            .sum::<Real>();
        0.5 * pe
    }
}

/// Methods for center-of-mass determination and adjustment.
impl ParticleSystem {
    /// Get center-of-mass mass (a.k.a. total mass).
    pub fn com_m(&self) -> Real {
        let mut msum = 0.0;
        for p in self.particles.iter() {
            msum += p.m;
        }
        msum
    }
    /// Get center-of-mass position.
    pub fn com_r(&self) -> [Real; 3] {
        let mut mtot = 0.0;
        let mut rcom = [0.0; 3];
        for p in self.particles.iter() {
            rcom[0] += p.m * p.r.0[0];
            rcom[1] += p.m * p.r.0[1];
            rcom[2] += p.m * p.r.0[2];
            mtot += p.m;
        }
        rcom.iter_mut().for_each(|r| *r /= mtot);
        rcom
    }
    /// Get center-of-mass velocity.
    pub fn com_v(&self) -> [Real; 3] {
        let mut mtot = 0.0;
        let mut vcom = [0.0; 3];
        for p in self.particles.iter() {
            vcom[0] += p.m * p.r.1[0];
            vcom[1] += p.m * p.r.1[1];
            vcom[2] += p.m * p.r.1[2];
            mtot += p.m;
        }
        vcom.iter_mut().for_each(|v| *v /= mtot);
        vcom
    }
    /// Moves center-of-mass to the given coordinates.
    pub fn com_move_to(&mut self, r: [Real; 3], v: [Real; 3]) {
        for p in self.particles.iter_mut() {
            p.r.0[0] += r[0];
            p.r.0[1] += r[1];
            p.r.0[2] += r[2];
            p.r.1[0] += v[0];
            p.r.1[1] += v[1];
            p.r.1[2] += v[2];
        }
    }
    /// Moves center-of-mass to the origin of coordinates.
    pub fn com_to_origin(&mut self) {
        let mut rcom = self.com_r();
        let mut vcom = self.com_v();
        rcom.iter_mut().for_each(|r| *r = -*r);
        vcom.iter_mut().for_each(|v| *v = -*v);
        self.com_move_to(rcom, vcom);
    }
}

impl ParticleSystem {
    pub fn scale_mass(&mut self) {
        let mtot = self.com_m();
        self.particles.iter_mut().for_each(|p| p.m /= mtot);
    }

    pub fn set_eps(&mut self, eps: Real) {
        let n = self.particles.len();
        let mmean = 1.0 / n as Real;
        for p in self.particles.iter_mut() {
            p.e2 = eps * eps * (p.m / mmean).powf(2.0 / 3.0);
        }
    }

    pub fn scale_to_standard(&mut self, virial_ratio: Real) {
        assert!(virial_ratio > 0.0);
        assert!(virial_ratio < 1.0);
        let ke = self.kinectic_energy();
        let pe = self.potential_energy();
        let v_scale = (-virial_ratio * pe / ke).sqrt();
        for p in self.particles.iter_mut() {
            p.r.1[0] *= v_scale;
            p.r.1[1] *= v_scale;
            p.r.1[2] *= v_scale;
        }
        let ke = -virial_ratio * pe;
        let te = ke + pe;
        let r_scale = te / -0.25;
        let v_scale = 1.0 / r_scale.sqrt();
        for p in self.particles.iter_mut() {
            p.r.0[0] *= r_scale;
            p.r.0[1] *= r_scale;
            p.r.0[2] *= r_scale;
            p.r.1[0] *= v_scale;
            p.r.1[1] *= v_scale;
            p.r.1[2] *= v_scale;
        }
    }
}

// -- end of file --
