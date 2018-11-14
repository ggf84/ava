// Typedef Real = f32/f64

#[cfg(any(feature = "f32", not(feature = "f64")))]
pub use std::f32::*;
#[cfg(any(feature = "f32", not(feature = "f64")))]
pub type Real = f32;

#[cfg(all(feature = "f64", not(feature = "f32")))]
pub use std::f64::*;
#[cfg(all(feature = "f64", not(feature = "f32")))]
pub type Real = f64;

// Extension traits for SoA data types

pub trait Len {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait AsSlice<'a>: Len {
    type Output;
    fn as_slice(&'a self) -> Self::Output;
    fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output);
}

pub trait AsSliceMut<'a>: Len {
    type Output;
    fn as_mut_slice(&'a mut self) -> Self::Output;
    fn split_at_mut(&'a mut self, mid: usize) -> (Self::Output, Self::Output);
}

// Macros for implementing SoA data types

macro_rules! impl_len {
    ($type_name:ty) => {
        impl Len for $type_name {
            fn len(&self) -> usize {
                self._len
            }
        }
    };
}

macro_rules! impl_to_vec {
    ($type_name:ty, $vec_name:ident, {$($field_name:ident),*}) => {
        impl $type_name {
            pub fn to_vec(&self) -> $vec_name {
                $vec_name {
                    _len: self.len(),
                    $(
                        $field_name: self.$field_name.to_vec(),
                    )*
                }
            }
        }
    }
}

macro_rules! impl_as_slice {
    ($type_name:ty, $slice_name:ident, {$($field_name:ident),*}) => {
        impl<'a> AsSlice<'a> for $type_name {
            type Output = $slice_name<'a>;

            fn as_slice(&'a self) -> Self::Output {
                $slice_name {
                    _len: self.len(),
                    $(
                        $field_name: &self.$field_name[..],
                    )*
                }
            }

            fn split_at(&'a self, mid: usize) -> (Self::Output, Self::Output) {
                let mut lo = $slice_name::default();
                let mut hi = $slice_name::default();
                let (lo_len, hi_len) = (mid, self.len() - mid);
                lo._len = lo_len;
                hi._len = hi_len;
                $(
                    if self.$field_name.len() >= mid {
                        let (_lo, _hi) = self.$field_name.split_at(mid);
                        lo.$field_name = _lo;
                        hi.$field_name = _hi;
                    }
                )*
                (lo, hi)
            }
        }
    }
}

macro_rules! impl_as_slice_mut {
    ($type_name:ty, $slice_mut_name:ident, {$($field_name:ident),*}) => {
        impl<'a> AsSliceMut<'a> for $type_name {
            type Output = $slice_mut_name<'a>;

            fn as_mut_slice(&'a mut self) -> Self::Output {
                $slice_mut_name {
                    _len: self.len(),
                    $(
                        $field_name: &mut self.$field_name[..],
                    )*
                }
            }

            fn split_at_mut(&'a mut self, mid: usize) -> (Self::Output, Self::Output) {
                let mut lo = $slice_mut_name::default();
                let mut hi = $slice_mut_name::default();
                let (lo_len, hi_len) = (mid, self.len() - mid);
                lo._len = lo_len;
                hi._len = hi_len;
                $(
                    if self.$field_name.len() >= mid {
                        let (_lo, _hi) = self.$field_name.split_at_mut(mid);
                        lo.$field_name = _lo;
                        hi.$field_name = _hi;
                    }
                )*
                (lo, hi)
            }
        }
    }
}

macro_rules! impl_struct_of_array {
    (
        $(#[$vec_meta:meta])* $vec_vis:vis $vec_name:ident,
        $(#[$slice_meta:meta])* $slice_vis:vis $slice_name:ident,
        $(#[$slice_mut_meta:meta])* $slice_mut_vis:vis $slice_mut_name:ident,
        {
            $($(#[$field_doc:meta])* $field_vis:vis $field_name:ident : $field_type:ty),* $(,)*
        }
    ) => {

        $(#[$vec_meta])*
        #[derive(Default)]
        $vec_vis struct $vec_name {
            pub(crate) _len: usize,
            $(
                $(
                    #[doc="Vec of"]
                    #[$field_doc]
                    #[doc="."]
                )*
                $field_vis $field_name: Vec<$field_type>,
            )*
        }

        $(#[$slice_meta])*
        #[derive(Default)]
        $slice_vis struct $slice_name<'a> {
            pub(crate) _len: usize,
            $(
                $(
                    #[doc="Slice of"]
                    #[$field_doc]
                    #[doc="."]
                )*
                $field_vis $field_name: &'a [$field_type],
            )*
        }

        $(#[$slice_mut_meta])*
        #[derive(Default)]
        $slice_mut_vis struct $slice_mut_name<'a> {
            pub(crate) _len: usize,
            $(
                $(
                    #[doc="SliceMut of"]
                    #[$field_doc]
                    #[doc="."]
                )*
                $field_vis $field_name: &'a mut [$field_type],
            )*
        }

        impl $vec_name {
            pub fn zeros(len: usize) -> Self {
                $vec_name {
                    _len: len,
                    $(
                        $field_name: vec![Default::default(); len],
                    )*
                }
            }

            pub fn apply_permutation(&mut self, indices: &[usize]) {
                let nact = indices.len();

                fn apply_slice<T: Copy>(indices: &[usize], vec: &mut [T]) {
                    let vec2 = vec.to_vec();
                    vec.iter_mut().zip(indices).for_each(|(v, &i)| *v = vec2[i]);
                }

                $(
                    if self.$field_name.len() >= nact {
                        apply_slice(&indices, &mut self.$field_name[..nact]);
                    }
                )*
            }
        }

        impl_len!($vec_name);
        // impl_to_vec!($vec_name, $vec_name, {$($field_name),*});
        impl_as_slice!($vec_name, $slice_name, {$($field_name),*});
        impl_as_slice_mut!($vec_name, $slice_mut_name, {$($field_name),*});

        impl_len!($slice_name<'_>);
        impl_to_vec!($slice_name<'_>, $vec_name, {$($field_name),*});
        impl_as_slice!($slice_name<'_>, $slice_name, {$($field_name),*});

        impl_len!($slice_mut_name<'_>);
        impl_to_vec!($slice_mut_name<'_>, $vec_name, {$($field_name),*});
        impl_as_slice_mut!($slice_mut_name<'_>, $slice_mut_name, {$($field_name),*});
    }
}

// -- end of file --
