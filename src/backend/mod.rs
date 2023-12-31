pub mod cpu;

use crate::tensor::{dtype::Dtype, shape::Shape};

pub trait Backend: 'static + core::fmt::Debug + Clone + Send + Sync {
    type Dtype: Dtype;
    // Implement a wrapper struct that contains the ptr to your buffer to avoid copying data in backend::ops
    type Buffer: 'static + core::fmt::Debug + Clone + Send + Sync;

    // load ops
    fn from(data: &[Self::Dtype]) -> Self;
    fn to_vec(&self) -> Vec<Self::Dtype>;
    fn empty(shape: &Shape) -> Self;
    fn const_like(&self, const_: Self::Dtype) -> Self;
    fn rand(shape: &Shape) -> Self;

    // binary
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
    fn bmax(&self, rhs: &Self) -> Self;
    fn cmplt(&self, rhs: &Self) -> Self;

    // unary
    fn log2(&self) -> Self;
    fn exp2(&self) -> Self;
    fn sin(&self) -> Self;
    fn sqrt(&self) -> Self;

    // reduce
    fn sum(&self, axis: Option<isize>, keepdim: bool) -> Self;
    fn max(&self, axis: Option<isize>, keepdim: bool) -> Self;

    // Ternary
    fn _where(&self, x: &Self, y: &Self) -> Self;

    // movement
    fn permute<S: Into<Shape>>(&self, permute: S) -> Self;
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Self;
    fn expand<S: Into<Shape>>(&self, shape: S) -> Self;
    fn shrink<A: Into<Vec<(usize, usize)>>>(&self, arg: A) -> Self;
    fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: Self::Dtype) -> Self;

    fn shape(&self) -> Shape;
    fn strides(&self) -> Shape;

    fn contiguous(&self) -> Self;

    fn col_i(&self, mut idx: usize) -> usize {
        let mut out = 0;
        for (sh, st) in self.shape().dims.iter().zip(self.strides().dims.iter()) {
            out += (idx % sh) * *st;
            idx /= sh;
        }
        out
    }

    fn row_i(&self, mut idx: usize) -> usize {
        let mut out = 0;
        for (sh, st) in self
            .shape()
            .dims
            .iter()
            .zip(self.strides().dims.iter())
            .rev()
        {
            out += (idx % sh) * *st;
            idx /= sh;
        }
        out
    }
}

fn strided_i(shape: &Shape, stride: &Shape, mut idx: usize) -> usize {
    let mut out = 0;
    for (sh, st) in shape.dims.iter().zip(stride.dims.iter()).rev() {
        out += (idx % sh) * *st;
        idx /= sh;
    }
    out
}
