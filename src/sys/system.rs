use crate::{real::Real, sys::particles::Particle};
use serde_derive::{Deserialize, Serialize};
use std::slice::{Iter, IterMut};

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
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
    pub fn iter(&self) -> Iter<'_, Particle> {
        self.particles.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<'_, Particle> {
        self.particles.iter_mut()
    }
    pub fn as_slice(&self) -> &[Particle] {
        self.particles.as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [Particle] {
        self.particles.as_mut_slice()
    }
    pub fn sort_by_dt(&mut self, n: usize) {
        self.particles[..n].sort_unstable_by(|a, b| (a.dt).partial_cmp(&b.dt).unwrap());
    }
    pub fn set_shared_dt(&mut self, dt: Real) {
        for p in self.iter_mut() {
            p.dt = dt;
        }
    }
}

/// Methods for center-of-mass determination and adjustment.
impl ParticleSystem {
    /// Get the total mass of the system.
    pub fn com_mass(&self) -> Real {
        let mut mtot = 0.0;
        for p in self.iter() {
            mtot += p.mass;
        }
        mtot
    }
    /// Get the total mass, position and velocity coordinates of the center-of-mass.
    pub fn com_mass_pos_vel(&self) -> (Real, [Real; 3], [Real; 3]) {
        let mut mtot = 0.0;
        let mut rcom = [0.0; 3];
        let mut vcom = [0.0; 3];
        for p in self.iter() {
            mtot += p.mass;
            for k in 0..3 {
                rcom[k] += p.mass * p.pos[k];
                vcom[k] += p.mass * p.vel[k];
            }
        }
        rcom.iter_mut().for_each(|r| *r /= mtot);
        vcom.iter_mut().for_each(|v| *v /= mtot);
        (mtot, rcom, vcom)
    }
    /// Moves the center-of-mass by the given position and velocity coordinates.
    pub fn com_move_by(&mut self, dpos: [Real; 3], dvel: [Real; 3]) {
        for p in self.iter_mut() {
            for k in 0..3 {
                p.pos[k] += dpos[k];
                p.vel[k] += dvel[k];
            }
        }
    }
}

impl ParticleSystem {
    /// Scale masses by a given factor.
    ///
    /// Returns the total mass after scaling.
    pub fn scale_mass(&mut self, m_scale: Real) -> Real {
        let mut mtot = 0.0;
        for p in self.iter_mut() {
            p.mass *= m_scale;
            mtot += p.mass;
        }
        mtot
    }
    /// Scale positions and velocities by a given factor.
    pub fn scale_pos_vel(&mut self, r_scale: Real) {
        let v_scale = 1.0 / r_scale.sqrt();
        for p in self.iter_mut() {
            for k in 0..3 {
                p.pos[k] *= r_scale;
                p.vel[k] *= v_scale;
            }
        }
    }
    /// Scale velocities to a given virial ratio.
    ///
    /// Returns the kinetic energy after the scaling.
    pub fn scale_to_virial(&mut self, q_vir: Real, ke: Real, pe: Real) -> Real {
        assert!(q_vir < 1.0);
        let v_scale = (-q_vir * pe / ke).sqrt();
        for p in self.iter_mut() {
            for k in 0..3 {
                p.vel[k] *= v_scale;
            }
        }
        -q_vir * pe
    }
    /// Scale positions and velocities to standard units (G = M = -4E = 1).
    pub fn scale_to_standard(&mut self, te0: Real, te: Real) {
        let r_scale = te / te0;
        self.scale_pos_vel(r_scale);
    }
    /// Set the softening parameter.
    pub fn set_eps(&mut self, eps_scale: Real) {
        let n = self.len();
        // let c = 2.0;
        // let eps = eps_scale * (c / n as Real);
        let c = (4.0 as Real / 27.0).sqrt();
        let eps = eps_scale * (c / n as Real).sqrt();
        for p in self.iter_mut() {
            p.eps = eps;
        }
    }
}

// -- end of file --
