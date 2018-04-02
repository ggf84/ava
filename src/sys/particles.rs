use real::Real;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Particle {
    /// Id of the particle
    pub id: usize,

    /// Mass of the particle
    pub m: Real,

    /// Squared softening of the particle
    pub e2: Real,

    /// Position and its derivatives
    pub r: ([Real; 3], [Real; 3], [Real; 3], [Real; 3]),
}

impl Particle {
    pub fn new(id: usize, m: Real, r: [Real; 3], v: [Real; 3]) -> Self {
        Particle {
            id: id,
            m: m,
            r: (r, v, [0.0; 3], [0.0; 3]),
            ..Default::default()
        }
    }
}

// -- end of file --
