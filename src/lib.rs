extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;

#[macro_use]
extern crate serde_derive;

pub mod compute;
pub mod ics;
pub mod sim;
pub mod sys;

pub mod real {
    #[cfg(any(feature = "f32", not(feature = "f64")))]
    pub use std::f32::*;
    #[cfg(any(feature = "f32", not(feature = "f64")))]
    pub type Real = f32;

    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub use std::f64::*;
    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub type Real = f64;
}

// -- end of file --
