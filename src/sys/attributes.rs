use crate::types::{AsSlice, AsSliceMut, Len, Real};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

impl_struct_of_array!(
    #[derive(Clone, Debug, PartialEq)]
    #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
    pub AttributesVec,
    pub AttributesSlice,
    pub AttributesSliceMut,
    {
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
);

// -- end of file --
