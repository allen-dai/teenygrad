pub mod view;

use core::ops::Index;
use num_traits::{PrimInt, ToPrimitive};

#[derive(Clone, Eq, PartialEq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl IntoIterator for Shape {
    type Item = usize;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl Shape {
    pub fn new<Dims: Into<Vec<usize>>>(dims: Dims) -> Self {
        let dims = dims.into();
        Self { dims }
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn strides(&self) -> Shape {
        let mut dims = vec![1; self.dims.len()];
        let mut stride = 1;
        dims.iter_mut()
            .zip(self.dims.iter())
            .rev()
            .for_each(|(st, sh)| {
                *st = stride;
                stride *= *sh
            });
        Shape { dims }
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }
}

impl<const D: usize, I: PrimInt + ToPrimitive> From<[I; D]> for Shape {
    fn from(value: [I; D]) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_usize().unwrap()).collect(),
        }
    }
}

impl<I: PrimInt + ToPrimitive> From<Vec<I>> for Shape {
    fn from(value: Vec<I>) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_usize().unwrap()).collect(),
        }
    }
}

impl<I: PrimInt + ToPrimitive> From<&[I]> for Shape {
    fn from(value: &[I]) -> Self {
        Self {
            dims: value.iter().map(|e| e.to_usize().unwrap()).collect(),
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl core::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl Index<isize> for Shape {
    type Output = usize;

    fn index(&self, index: isize) -> &Self::Output {
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.dims[index]
    }
}

impl core::ops::IndexMut<isize> for Shape {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.dims[index]
    }
}

impl Index<i32> for Shape {
    type Output = usize;

    fn index(&self, index: i32) -> &Self::Output {
        let index = if index < 0 {
            (self.len() as i32 + index) as usize
        } else {
            index as usize
        };
        &self.dims[index]
    }
}

impl core::ops::IndexMut<i32> for Shape {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        let index = if index < 0 {
            (self.len() as i32 + index) as usize
        } else {
            index as usize
        };
        &mut self.dims[index]
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

impl core::fmt::Debug for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}
