use real::Real;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Particle {
    /// Id of the particle
    pub id: usize,

    /// Softening of the particle
    pub eps: Real,

    /// Mass of the particle
    pub mass: Real,

    /// xyz-Position
    pub pos: [Real; 3],

    /// xyz-Velocity
    pub vel: [Real; 3],

    /// xyz-Acceleration's 0-derivative
    pub acc0: [Real; 3],

    /// xyz-Acceleration's 1-derivative
    pub acc1: [Real; 3],

    /// xyz-Acceleration's 2-derivative
    pub acc2: [Real; 3],

    /// xyz-Acceleration's 3-derivative
    pub acc3: [Real; 3],
}

impl Particle {
    pub fn new(id: usize, mass: Real, pos: [Real; 3], vel: [Real; 3]) -> Self {
        Particle {
            id: id,
            mass: mass,
            pos: pos,
            vel: vel,
            ..Default::default()
        }
    }
}

// -- end of file --
