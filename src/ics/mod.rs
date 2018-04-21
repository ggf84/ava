use rand::Rng;
use real::{consts, Real};
use sys::particles::Particle;
use sys::system::ParticleSystem;

pub mod imf;
pub mod sdp;

pub trait Sampler {
    type Output;
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Output;
}

pub struct Model<IMF, SDP> {
    imf: IMF,
    sdp: SDP,
}

impl<IMF, SDP> Model<IMF, SDP>
where
    IMF: Sampler<Output = Real>,
    SDP: Sampler<Output = (Real, Real)>,
{
    pub fn new(imf: IMF, sdp: SDP) -> Self {
        Model { imf: imf, sdp: sdp }
    }

    pub fn build<R: Rng>(&self, n: usize, rng: &mut R) -> ParticleSystem {
        let mut ps = ParticleSystem::new();
        for id in 0..n {
            let m = self.imf.sample(rng);
            let (r, v) = self.sdp.sample(rng);
            let particle = Particle::new(id, m, to_xyz(r, rng), to_xyz(v, rng));
            ps.particles.push(particle);
        }
        ps.scale_mass();
        ps.com_to_origin();
        ps.scale_to_standard(0.5);
        ps
    }
}

fn to_xyz<R: Rng>(val: Real, rng: &mut R) -> [Real; 3] {
    let theta = rng.gen_range::<Real>(-1.0, 1.0).acos();
    let phi = rng.gen_range::<Real>(0.0, 2.0 * consts::PI);
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    [val * st * cp, val * st * sp, val * ct]
}

// -- end of file --
