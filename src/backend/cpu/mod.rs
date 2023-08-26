use std::{collections::VecDeque, sync::Arc};

use super::Backend;
use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct CpuBuffer<T: Dtype = f32>(VecDeque<T>);

pub struct Cpu<T: Dtype = f32> {
    buffer: Arc<CpuBuffer<T>>,
    shape: Shape,
    stride: Shape,
}

unsafe impl<T: Dtype> Send for Cpu<T> {}
unsafe impl<T: Dtype> Sync for Cpu<T> {}

impl<T: Dtype> Clone for Cpu<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
        }
    }
}

impl<T: Dtype> core::fmt::Debug for Cpu<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.shape.numel() {
            //write!(f, "{:<12?}", self_idx);
            write!(f, "{:>12?}", self.buffer.0[self.row_i(i)]);
            if (i + 1) % self.shape[self.shape.len() - 1] == 0 {
                write!(f, "\n");
            }

            if (i + 1)
                % self.shape.dims[self.shape.len() - 2..]
                    .iter()
                    .product::<usize>()
                == 0
            {
                write!(f, "\n");
            }
        }
        Ok(())
    }
}

impl<T: Dtype> core::ops::Deref for Cpu<T> {
    type Target = VecDeque<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer.0
    }
}

impl<T: Dtype> core::ops::DerefMut for Cpu<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut Arc::get_mut(&mut self.buffer)
            .expect("Only one mutable reference")
            .0
    }
}

impl<T: Dtype> Backend for Cpu<T> {
    type Dtype = T;
    type Buffer = CpuBuffer<T>;

    fn from_vec(data: Vec<Self::Dtype>, shape: &Shape) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(data.into())),
            shape: shape.clone(),
            stride: shape.strides(),
        }
    }

    fn to_vec(&self) -> Vec<T> {
        self.buffer.0.clone().into()
    }

    fn empty(shape: &Shape) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(VecDeque::new())),
            shape: shape.clone(),
            stride: shape.strides(),
        }
    }

    fn const_like(&self, const_: Self::Dtype) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(vec![const_; self.shape.numel()].into())),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn rand<D: rand_distr::Distribution<Self::Dtype>>(shape: &Shape, dist: D) -> Self {
        let mut rng = crate::RNG.lock().unwrap();
        let mut out = vec![Default::default(); shape.numel()];
        for i in 0..shape.numel() {
            out[i] = rng.sample(&dist);
        }
        Cpu {
            buffer: Arc::new(CpuBuffer(out.into())),
            shape: shape.clone(),
            stride: shape.strides(),
        }
    }

    fn add(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)] + rhs.buffer.0[rhs.row_i(i)])
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)] - rhs.buffer.0[rhs.row_i(i)])
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)] * rhs.buffer.0[rhs.row_i(i)])
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn div(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)] / rhs.buffer.0[rhs.row_i(i)])
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn bmax(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| {
                        if self.buffer.0[self.row_i(i)] >= rhs.buffer.0[rhs.row_i(i)] {
                            self.buffer.0[self.row_i(i)]
                        } else {
                            rhs.buffer.0[rhs.row_i(i)]
                        }
                    })
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn cmplt(&self, rhs: &Self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| {
                        if self.buffer.0[self.row_i(i)] < rhs.buffer.0[rhs.row_i(i)] {
                            self.buffer.0[self.row_i(i)]
                        } else {
                            rhs.buffer.0[rhs.row_i(i)]
                        }
                    })
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn log2(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)].log2())
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn exp2(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)].exp2())
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn sin(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)].sin())
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn sqrt(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)].sqrt())
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn sum(&self, _axis: Option<isize>) -> Self {
        if _axis.is_none() {
            return Cpu {
                buffer: Arc::new(CpuBuffer(
                    vec![self.buffer.0.iter().fold(T::zero(), |acc, x| acc + *x)].into(),
                )),
                shape: Shape::from([1]),
                stride: Shape::from([1]),
            };
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
                out[idx] += self.buffer.0[self.row_i(i)];
                idx += 1;
            } else {
                out[idx] += self.buffer.0[self.row_i(i)];
                if (i + 1) % numel_of_reduce_dim == 0 {
                    idx += 1;
                }
            }
        }
        out
    }

    fn max(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                vec![self
                    .buffer
                    .0
                    .iter()
                    .fold(T::min_value(), |acc, e| if acc > *e { acc } else { *e })]
                .into(),
            )),
            shape: Shape::from([1]),
            stride: Shape::from([1]),
        }
    }

    fn permute<S: Into<Shape>>(&self, permute: S) -> Self {
        let mut out = self.clone();
        let mut permute = permute.into();
        assert!(
            permute.dims.iter().max().unwrap() < &out.shape.len(),
            "Permute index cannot be be >= number of dims: P: {} Dim: {}",
            permute.dims.iter().max().unwrap(),
            out.shape.len()
        );
        for i in 0..permute.len() {
            let mut pi = permute[i];
            while pi < i {
                pi = permute[pi];
            }
            out.shape.dims.as_mut_slice().swap(pi, i);
            out.stride.dims.as_mut_slice().swap(pi, i);
        }
        out
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> Self {
        let shape = shape.into();
        let mut out = self.clone();
        out.shape = shape.clone();
        out.stride = shape.strides();
        if self.buffer.0.len() != self.shape().numel() {
            println!("reshape copied mem");
            out.buffer = Self::contiguous(&self).buffer.clone();
        }
        out
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> Self {
        let shape = shape.into();
        let mut out = self.clone();
        out.shape
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
        out.shape = shape.clone();
        out.stride = Shape::from(new_stride);
        out
    }

    fn shrink<A: Into<Vec<(usize, usize)>>>(&self, arg: A) -> Self {
        let mut arg: Vec<(usize, usize)> = arg.into();
        assert!(arg.len() == self.shape.len() || arg.len() == 1);
        if arg.len() == 1 && self.shape.len() != 1 {
            arg.repeat(self.shape().len());
        }
        let new_shape = Shape::from(
            self.shape
                .dims
                .iter()
                .zip(arg.iter())
                .map(|(dim, aarg)| aarg.0.abs_diff(aarg.1))
                .collect::<Vec<usize>>(),
        );
        let mut out = Self::empty(&new_shape).const_like(T::zero());
        let mut iter_shape = vec![0usize; self.shape.len()];
        let mut self_idx = 0;

        for i in 0..self.shape.numel() {
            if iter_shape.iter().enumerate().all(|(ii, iter_d)| {
                if *iter_d >= arg[ii].0 && *iter_d < arg[ii].1 {
                    true
                } else {
                    false
                }
            }) {
                // println!("{:?}", iter_shape);
                out[self_idx] = self[self.row_i(i)];
                self_idx += 1;
            }
            *iter_shape.last_mut().unwrap() += 1;
            let mut tmp = 0;
            iter_shape
                .iter_mut()
                .zip(self.shape.dims.iter())
                .rev()
                .for_each(|(ish, nsh)| {
                    *ish += tmp;
                    if *ish >= *nsh {
                        tmp = 1;
                        *ish = 0;
                    } else {
                        tmp = 0;
                    }
                });
        }
        out
    }

    fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: Self::Dtype) -> Self {
        let mut arg: Vec<(usize, usize)> = arg.into();
        assert!(arg.len() == self.shape.len() || arg.len() == 1);
        if arg.len() == 1 && self.shape.len() != 1 {
            arg.repeat(self.shape().len());
        }
        let new_shape = Shape::from(
            self.shape
                .dims
                .iter()
                .zip(arg.iter())
                .map(|(dim, aarg)| dim + aarg.0 + aarg.1)
                .collect::<Vec<usize>>(),
        );
        let mut out = Self::empty(&new_shape).const_like(const_value);
        let mut iter_shape = vec![0usize; new_shape.len()];
        let mut self_idx = 0;

        for i in 0..out.shape.numel() {
            if iter_shape.iter().enumerate().all(|(ii, d)| {
                if *d >= arg[ii].0 && *d < arg[ii].0 + self.shape.dims[ii] {
                    true
                } else {
                    false
                }
            }) {
                // println!("{:?} {i}", iter_shape);
                out[i] = self[self.row_i(self_idx)];
                self_idx += 1;
            }

            *iter_shape.last_mut().unwrap() += 1;
            let mut tmp = 0;
            iter_shape
                .iter_mut()
                .zip(new_shape.dims.iter())
                .rev()
                .for_each(|(ish, nsh)| {
                    *ish += tmp;
                    if *ish >= *nsh {
                        tmp = 1;
                        *ish = 0;
                    } else {
                        tmp = 0;
                    }
                });
        }
        out
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn stride(&self) -> Shape {
        self.stride.clone()
    }

    fn contiguous(&self) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| self.buffer.0[self.row_i(i)])
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    // fn raw_ptr(&self) -> *const Self::Buffer {
    //     &*(self.buffer) as *const Self::Buffer
    // }
    //
    // fn set_stride<S: Into<Shape>>(&mut self, shape: S) {
    //     self.stride = shape.into();
    // }
}

#[test]
fn cpu_pad() {
    use crate::prelude::*;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..=3 * 3 * 3)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [3, 3, 3],
    );
    let r = t.pad([(1, 1), (1, 1), (1, 1)], 0f32).slice([(1,9),(1,9),(1,9)], 0f32);
    // let rr = r.shrink([(1, 2), (1, 2), (1, 2)]);
    //println!("{t:?}");
    println!("{:?} {} {}", r, r.shape(), r.stride());
    //println!("{:?} {} {}", rr, rr.shape(), rr.stride());
}
