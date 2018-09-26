use crate::real::Real;
use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    /// Id of the particle
    pub id: u64,

    /// Time-step of the particle
    pub dt: Real,

    /// Current time of the particle
    pub tnow: Real,

    /// Softening of the particle
    pub eps: Real,

    /// Mass of the particle
    pub mass: Real,

    /// Position
    pub pos: [Real; 3],

    /// Velocity
    pub vel: [Real; 3],

    /// Acceleration's 0-derivative
    pub acc0: [Real; 3],

    /// Acceleration's 1-derivative
    pub acc1: [Real; 3],

    /// Acceleration's 2-derivative
    pub acc2: [Real; 3],

    /// Acceleration's 3-derivative
    pub acc3: [Real; 3],

    /// Acceleration's 4-derivative
    pub acc4: [Real; 3],

    /// Acceleration's 5-derivative
    pub acc5: [Real; 3],
}

impl Particle {
    pub fn new(id: u64, mass: Real, pos: [Real; 3], vel: [Real; 3]) -> Self {
        Particle {
            id,
            mass,
            pos,
            vel,
            ..Default::default()
        }
    }
}

// -- end of file --
