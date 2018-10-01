use crate::real::Real;
use serde::ser::{self, SerializeStruct};
use serde_derive::{Deserialize, Serialize};
use soa_derive::StructOfArray;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize, StructOfArray)]
#[soa_derive = "Clone, Debug, PartialEq, Deserialize"]
pub struct Attributes {
    /// Id
    pub id: u64,

    /// Time-step
    pub dt: Real,

    /// Softening
    pub eps: Real,

    /// Mass
    pub mass: Real,

    /// Current time
    pub tnow: Real,

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

    /// New Current time
    pub new_tnow: Real,

    /// New Position
    pub new_pos: [Real; 3],

    /// New Velocity
    pub new_vel: [Real; 3],

    /// New Acceleration's 0-derivative
    pub new_acc0: [Real; 3],

    /// New Acceleration's 1-derivative
    pub new_acc1: [Real; 3],

    /// New Acceleration's 2-derivative
    pub new_acc2: [Real; 3],

    /// New Acceleration's 3-derivative
    pub new_acc3: [Real; 3],

    /// New Acceleration's 4-derivative
    pub new_acc4: [Real; 3],

    /// New Acceleration's 5-derivative
    pub new_acc5: [Real; 3],
}

impl Default for AttributesVec {
    fn default() -> Self {
        Self::new()
    }
}

impl ser::Serialize for AttributesVec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        // 22 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("AttributesVec", 22)?;

        state.serialize_field("id", &self.id)?;
        state.serialize_field("dt", &self.dt)?;
        state.serialize_field("eps", &self.eps)?;
        state.serialize_field("mass", &self.mass)?;

        state.serialize_field("tnow", &self.tnow)?;
        state.serialize_field("pos", &self.pos)?;
        state.serialize_field("vel", &self.vel)?;
        state.serialize_field("acc0", &self.acc0)?;
        state.serialize_field("acc1", &self.acc1)?;
        state.serialize_field("acc2", &self.acc2)?;
        state.serialize_field("acc3", &self.acc3)?;
        state.serialize_field("acc4", &self.acc4)?;
        state.serialize_field("acc5", &self.acc5)?;

        state.serialize_field("new_tnow", &self.new_tnow)?;
        state.serialize_field("new_pos", &self.new_pos)?;
        state.serialize_field("new_vel", &self.new_vel)?;
        state.serialize_field("new_acc0", &self.new_acc0)?;
        state.serialize_field("new_acc1", &self.new_acc1)?;
        state.serialize_field("new_acc2", &self.new_acc2)?;
        state.serialize_field("new_acc3", &self.new_acc3)?;
        state.serialize_field("new_acc4", &self.new_acc4)?;
        state.serialize_field("new_acc5", &self.new_acc5)?;

        state.end()
    }
}

impl AttributesVec {
    pub fn apply_permutation(&mut self, indices: &[usize]) {
        let nact = indices.len();

        fn apply_slice<T: Copy>(indices: &[usize], vec: &mut [T]) {
            let vec2 = vec.to_vec();
            vec.iter_mut().zip(indices).for_each(|(v, &i)| *v = vec2[i]);
        }

        apply_slice(&indices, &mut self.id[..nact]);
        apply_slice(&indices, &mut self.dt[..nact]);
        apply_slice(&indices, &mut self.eps[..nact]);
        apply_slice(&indices, &mut self.mass[..nact]);

        apply_slice(&indices, &mut self.tnow[..nact]);
        apply_slice(&indices, &mut self.pos[..nact]);
        apply_slice(&indices, &mut self.vel[..nact]);
        apply_slice(&indices, &mut self.acc0[..nact]);
        apply_slice(&indices, &mut self.acc1[..nact]);
        apply_slice(&indices, &mut self.acc2[..nact]);
        apply_slice(&indices, &mut self.acc3[..nact]);
        apply_slice(&indices, &mut self.acc4[..nact]);
        apply_slice(&indices, &mut self.acc5[..nact]);

        apply_slice(&indices, &mut self.new_tnow[..nact]);
        apply_slice(&indices, &mut self.new_pos[..nact]);
        apply_slice(&indices, &mut self.new_vel[..nact]);
        apply_slice(&indices, &mut self.new_acc0[..nact]);
        apply_slice(&indices, &mut self.new_acc1[..nact]);
        apply_slice(&indices, &mut self.new_acc2[..nact]);
        apply_slice(&indices, &mut self.new_acc3[..nact]);
        apply_slice(&indices, &mut self.new_acc4[..nact]);
        apply_slice(&indices, &mut self.new_acc5[..nact]);

        // fn apply_slice_unsafe(indices: &[usize], mut vec: AttributesSliceMut) {
        //     let mut vec2 = vec.to_vec();
        //     let mut vec2 = vec2.as_mut_slice();
        //     vec.iter_mut().zip(indices).for_each(|(mut v, &i)| {
        //         unsafe {
        //             let mut v2 = vec2.get_unchecked_mut(i);
        //             v.as_mut_ptr().write_unaligned(v2.as_mut_ptr().read_unaligned());
        //         }
        //     });
        // }
        // apply_slice_unsafe(&indices, self.slice_mut(0..nact));
    }
}

// -- end of file --
