extern crate rand;
extern crate rayon;

pub mod compute;
pub mod ics;
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

    #[repr(align(16))]
    #[derive(Debug, Default, Copy, Clone)]
    pub struct Aligned<T>(T);
    impl<T> ::std::ops::Deref for Aligned<T> {
        type Target = T;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T> ::std::ops::DerefMut for Aligned<T> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
}

// -- end of file --
