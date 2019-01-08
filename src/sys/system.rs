use crate::{
    gravity::{
        energy::{Energy, EnergyKernel},
        Compute,
    },
    sys::AttributesVec,
    types::{AsSlice, AsSliceMut, Len, Real},
};
use itertools::izip;
use rand::{distributions::Distribution, Rng};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ParticleSystem {
    pub time: Real,
    pub attrs: AttributesVec,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Len for ParticleSystem {
    fn len(&self) -> usize {
        self.attrs.len()
    }
}

impl ParticleSystem {
    pub fn from_model<Model, R>(npart: usize, model: &Model, rng: &mut R) -> Self
    where
        Model: Distribution<(Real, [Real; 3], [Real; 3])>,
        R: Rng,
    {
        let mut attrs = AttributesVec::default();
        attrs._len = npart;
        attrs.id = Vec::with_capacity(npart);
        attrs.mass = Vec::with_capacity(npart);
        attrs.pos = Vec::with_capacity(npart);
        attrs.vel = Vec::with_capacity(npart);
        attrs.eps = vec![Default::default(); npart];
        for (i, (m, r, v)) in model.sample_iter(rng).take(npart).enumerate() {
            attrs.id.push(i as u64);
            attrs.mass.push(m);
            attrs.pos.push(r);
            attrs.vel.push(v);
        }
        // let mut attrs = AttributesVec::zeros(npart);
        // for (i, (m, r, v)) in model.sample_iter(rng).take(npart).enumerate() {
        //     attrs.id[i] = i as u64;
        //     attrs.mass[i] = m;
        //     attrs.pos[i] = r;
        //     attrs.vel[i] = v;
        // }
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

        let mut energy = Energy::zeros(self.len());
        EnergyKernel {}.compute(self.attrs.as_slice().into(), energy.as_mut_slice());
        let (ke, pe) = energy.reduce(mtot);
        let ke = self.scale_to_virial(q_vir, ke, pe);
        self.scale_pos_vel((ke + pe) / -0.25);

        let mut energy = Energy::zeros(self.len());
        EnergyKernel {}.compute(self.attrs.as_slice().into(), energy.as_mut_slice());
        let (ke, pe) = energy.reduce(mtot);
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
        for (&m, &r, &v) in izip!(&self.attrs.mass, &self.attrs.pos, &self.attrs.vel) {
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
        for (r, v) in izip!(&mut self.attrs.pos, &mut self.attrs.vel) {
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
