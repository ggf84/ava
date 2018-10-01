use crate::{
    gravity::Energy,
    real::Real,
    sys::attributes::{Attributes, AttributesVec},
};
use rand::{distributions::Distribution, Rng};
use serde_derive::{Deserialize, Serialize};
use soa_derive::{soa_zip, soa_zip_impl};
use std::convert::{AsMut, AsRef};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleSystem {
    pub time: Real,
    pub attrs: AttributesVec,
}

impl AsRef<Self> for ParticleSystem {
    fn as_ref(&self) -> &Self {
        self
    }
}
impl AsMut<Self> for ParticleSystem {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl ParticleSystem {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn len(&self) -> usize {
        self.attrs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
}

impl ParticleSystem {
    pub fn from_model<Model, R>(npart: usize, model: &Model, rng: &mut R) -> Self
    where
        Model: Distribution<(Real, [Real; 3], [Real; 3])>,
        R: Rng,
    {
        let mut attrs = AttributesVec::with_capacity(npart);
        for (i, (m, r, v)) in model.sample_iter(rng).take(npart).enumerate() {
            let attr = Attributes {
                id: i as u64,
                mass: m,
                pos: r,
                vel: v,
                ..Default::default()
            };
            attrs.push(attr);
        }
        ParticleSystem { time: 0.0, attrs }
    }
    /// Convert the system to standard units (G = M = -4E = 1).
    pub fn into_standard_units(mut self, q_vir: Real, eps_param: Option<Real>) -> Self {
        self.convert_to_standard_units(q_vir, eps_param);
        self
    }
    fn convert_to_standard_units(&mut self, q_vir: Real, eps_param: Option<Real>) {
        let (mtot, [rx, ry, rz], [vx, vy, vz]) = self.com_mass_pos_vel();
        // reset center-of-mass to the origin of coordinates.
        self.com_move_by([-rx, -ry, -rz], [-vx, -vy, -vz]);
        let mtot = self.scale_mass(1.0 / mtot);

        if let Some(eps_param) = eps_param {
            self.set_eps(eps_param * 1.0); // assume rvir == 1
        }

        let (ke, pe) = Energy::new(mtot).energies(self.as_ref());
        let ke = self.scale_to_virial(q_vir, ke, pe);
        self.scale_pos_vel((ke + pe) / -0.25);

        let (ke, pe) = Energy::new(mtot).energies(self.as_ref());
        let rvir = mtot.powi(2) / (-2.0 * pe);
        eprintln!(
            "mtot: {:?}\nke: {:?}\npe: {:?}\nte: {:?}\nrvir: {:?}",
            mtot,
            ke,
            pe,
            ke + pe,
            rvir
        );
    }
    /// Set the softening parameter.
    fn set_eps(&mut self, eps_scale: Real) {
        let n = self.len();
        // let c = 2.0;
        // let value = eps_scale * (c / n as Real);
        let c = (4.0 as Real / 27.0).sqrt();
        let value = eps_scale * (c / n as Real).sqrt();
        self.attrs.eps.iter_mut().for_each(|e| *e = value);
    }
}

impl ParticleSystem {
    /// Get the total mass of the system.
    pub fn com_mass(&self) -> Real {
        self.attrs.mass.iter().sum()
    }
    /// Get the total mass, position and velocity coordinates of the center-of-mass.
    pub fn com_mass_pos_vel(&self) -> (Real, [Real; 3], [Real; 3]) {
        let mtot = self.com_mass();
        let mut rcom = [0.0; 3];
        let mut vcom = [0.0; 3];
        for (&m, &r, &v) in soa_zip!(&self.attrs, [mass, pos, vel]) {
            rcom.iter_mut()
                .zip(&r)
                .for_each(|(rcom, &r)| *rcom += m * r);
            vcom.iter_mut()
                .zip(&v)
                .for_each(|(vcom, &v)| *vcom += m * v);
        }
        rcom.iter_mut().for_each(|r| *r /= mtot);
        vcom.iter_mut().for_each(|v| *v /= mtot);
        (mtot, rcom, vcom)
    }
    /// Moves the center-of-mass by the given position and velocity displacements.
    pub fn com_move_by(&mut self, dr: [Real; 3], dv: [Real; 3]) {
        for (r, v) in soa_zip!(&mut self.attrs, [mut pos, mut vel]) {
            r.iter_mut().zip(&dr).for_each(|(r, &dr)| *r += dr);
            v.iter_mut().zip(&dv).for_each(|(v, &dv)| *v += dv);
        }
    }
}

impl ParticleSystem {
    /// Scale masses by m_scale. Returns the total mass after scaling.
    pub fn scale_mass(&mut self, m_scale: Real) -> Real {
        self.attrs.mass.iter_mut().fold(0.0, |sum, m| {
            *m *= m_scale;
            sum + *m
        })
    }
    /// Scale positions and velocities by r_scale (v_scale is computed from r_scale).
    pub fn scale_pos_vel(&mut self, r_scale: Real) {
        let v_scale = 1.0 / r_scale.sqrt();
        self.attrs
            .pos
            .iter_mut()
            .flatten()
            .for_each(|r| *r *= r_scale);
        self.attrs
            .vel
            .iter_mut()
            .flatten()
            .for_each(|v| *v *= v_scale);
    }
    /// Scale velocities to a given virial ratio. Returns the kinetic energy after the scaling.
    pub fn scale_to_virial(&mut self, q_vir: Real, ke: Real, pe: Real) -> Real {
        assert!(q_vir < 1.0);
        let v_scale = (-q_vir * pe / ke).sqrt();
        self.attrs
            .vel
            .iter_mut()
            .flatten()
            .for_each(|v| *v *= v_scale);
        -q_vir * pe
    }
}

// -- end of file --
