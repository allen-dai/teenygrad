use std::sync::Arc;

use super::Backend;
use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct CpuVec<T: Dtype = f32> {
    buffer: Vec<T>,
    shape: Shape,
    stride: Shape,
}

#[derive(Clone)]
pub struct Cpu<T: Dtype = f32>(Arc<CpuVec<T>>);
unsafe impl<T: Dtype> Send for Cpu<T> {}
unsafe impl<T: Dtype> Sync for Cpu<T> {}

impl<T: Dtype> core::fmt::Debug for Cpu<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.shape.numel() {
            //write!(f, "{:<12?}", self_idx);
            write!(f, "{:<12?}", self.buffer[self.row_i(i)]);
            if (i + 1) % self.shape[self.shape.len() - 1] == 0 {
                write!(f, "\n");
            }
        }
        Ok(())
    }
}

impl<T: Dtype> core::ops::Deref for Cpu<T> {
    type Target = CpuVec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Dtype> core::ops::DerefMut for Cpu<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::get_mut(&mut self.0).expect("Only one mutable reference")
    }
}

impl<T: Dtype> Backend for Cpu<T> {
    type Dtype = T;
    type Buffer = CpuVec<T>;

    fn from_vec(data: Vec<Self::Dtype>, shape: &Shape) -> Self {
        Cpu(Arc::new(CpuVec {
            buffer: data,
            shape: shape.clone(),
            stride: shape.strides(),
        }))
    }

    fn to_vec(&self) -> Vec<T> {
        self.buffer.to_vec()
    }

    fn empty(shape: &Shape) -> Self {
        Cpu(Arc::new(CpuVec {
            buffer: vec![],
            shape: shape.clone(),
            stride: shape.strides(),
        }))
    }

    fn const_like(&self, const_: Self::Dtype) -> Self {
        Cpu(Arc::new(CpuVec {
            buffer: vec![const_; self.shape.numel()],
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn rand<D: rand_distr::Distribution<Self::Dtype>>(shape: &Shape, dist: D) -> Self {
        let mut rng = crate::RNG.lock().unwrap();
        let mut out = vec![Default::default(); shape.numel()];
        for i in 0..shape.numel() {
            out[i] = rng.sample(&dist);
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: shape.clone(),
            stride: shape.strides(),
        }))
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)] + rhs.buffer[rhs.row_i(i)];
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)] - rhs.buffer[rhs.row_i(i)];
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn mul(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)] * rhs.buffer[rhs.row_i(i)];
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn div(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)] / rhs.buffer[rhs.row_i(i)];
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn bmax(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = {
                let mut out = T::zero();
                if self.buffer[self.row_i(i)] >= rhs.buffer[rhs.row_i(i)] {
                    out = self.buffer[self.row_i(i)];
                } else {
                    out = rhs.buffer[rhs.row_i(i)];
                }
                out
            }
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn cmplt(&self, rhs: &Self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = {
                let mut out = T::zero();
                if self.buffer[self.row_i(i)] < rhs.buffer[rhs.row_i(i)] {
                    out = self.buffer[self.row_i(i)];
                } else {
                    out = rhs.buffer[rhs.row_i(i)];
                }
                out
            }
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn log2(&self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)].log2();
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn exp2(&self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)].exp2();
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn sin(&self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)].sin();
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn sqrt(&self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)].sqrt();
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn sum(&self, _axis: Option<isize>) -> Self {
        if _axis.is_none() {
            return Cpu(Arc::new(CpuVec {
                buffer: vec![self.buffer.iter().fold(T::zero(), |acc, x| acc + *x)],
                shape: Shape::from([1]),
                stride: Shape::from([1]),
            }));
        }
        let _axis = _axis.unwrap();
        let mut axis = if _axis < 0 {
            (self.shape.len() as isize + _axis) as usize
        } else {
            _axis as usize
        };
        let mut new_shape = self.shape().clone();
        new_shape.dims.remove(axis);
        let mut out = Self::empty(&new_shape).const_like(T::zero());
        let numel_of_reduce_dim = if axis == self.shape.len() - 1 {
            *self.shape.dims.last().unwrap()
        } else {
            self.shape.dims[axis + 1..].iter().product::<usize>()
        };
        let mut dim_reduced = 0;
        let mut base = 0;
        let mut idx = 0;
        for i in 0..self.shape().numel() {
            if axis < self.shape.len() - 1 {
                if idx - base >= numel_of_reduce_dim {
                    idx = base;
                    dim_reduced += 1;
                };
                if dim_reduced == self.shape[axis] {
                    dim_reduced = 0;
                    base += numel_of_reduce_dim;
                    idx = base;
                }
                out.buffer[idx] += self.buffer[self.row_i(i)];
                idx += 1;
            } else {
                out.buffer[idx] += self.buffer[self.row_i(i)];
                if (i+1) % numel_of_reduce_dim == 0 {
                    idx += 1;
                }
            }
        }
        out
    }

    fn max(&self) -> Self {
        Cpu(Arc::new(CpuVec {
            buffer: vec![self
                .buffer
                .iter()
                .fold(T::min_value(), |acc, e| if acc > *e { acc } else { *e })],
            shape: Shape::from([1]),
            stride: Shape::from([1]),
        }))
    }

    fn permute<S: Into<Shape>>(&mut self, permute: S) -> Self {
        let mut permute = permute.into();
        assert!(
            permute.dims.iter().max().unwrap() < &self.shape.len(),
            "Permute index cannot be be >= number of dims: P: {} Dim: {}",
            permute.dims.iter().max().unwrap(),
            self.shape.len()
        );
        for i in 0..permute.len() {
            let mut pi = permute[i];
            while pi < i {
                pi = permute[pi];
            }
            self.shape.dims.as_mut_slice().swap(pi, i);
            self.stride.dims.as_mut_slice().swap(pi, i);
        }
        self.clone()
    }

    fn reshape<S: Into<Shape>>(&mut self, shape: S) -> Self {
        let shape = shape.into();
        self.shape = shape.clone();
        self.stride = shape.strides();
        if self.buffer.len() != self.shape().numel() {
            self.buffer = Self::contiguous(&self).buffer.clone();
        }
        self.clone()
    }

    fn expand<S: Into<Shape>>(&mut self, shape: S) -> Self {
        let shape = shape.into();
        self.shape
            .dims
            .iter()
            .rev()
            .zip(shape.dims.iter().rev())
            .enumerate()
            .for_each(|(i, (l, r))| {
                if l != r && *l != 1 {
                    panic!("dim {i} != expand shape dim {i} && dim {i} != 1")
                }
            });

        let mut new_stride = self
            .shape
            .dims
            .iter()
            .rev()
            .zip(self.stride.dims.iter().rev())
            .map(|(sh, st)| if *sh == 1 { 0 } else { *st })
            .collect::<Vec<usize>>();
        for _ in 0..shape.len() - self.shape.len() {
            new_stride.push(0);
        }
        new_stride.reverse();
        self.shape = shape.clone();
        self.stride = Shape::from(new_stride);
        self.clone()
    }

    fn shrink<S: Into<Shape>>(&mut self, shape: S) -> Self {
        todo!()
    }

    fn pad<S: Into<Shape>>(&mut self, arg: S) -> Self {
        todo!()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn stride(&self) -> Shape {
        self.stride.clone()
    }

    fn contiguous(&self) -> Self {
        let mut out = vec![T::zero(); self.shape.numel()];
        for i in 0..self.shape.numel() {
            out[i] = self.buffer[self.row_i(i)];
        }
        Cpu(Arc::new(CpuVec {
            buffer: out,
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }))
    }

    fn raw_ptr(&self) -> *const Self::Buffer {
        &*(self.0) as *const Self::Buffer
    }
}
