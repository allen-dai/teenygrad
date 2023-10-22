use std::{collections::VecDeque, sync::Arc};

use super::Backend;
use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct CpuBuffer<T: Dtype = f32>(Vec<T>);

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
            write!(f, "{:<14?}", self.buffer.0[self.row_i(i)])?;
            if (i + 1) % self.shape[self.shape.len() - 1] == 0 {
                write!(f, "\n")?;
            }

            if self.shape.len() >= 2
                && (i + 1)
                    % self.shape.dims[self.shape.len() - 2..]
                        .iter()
                        .product::<usize>()
                    == 0
            {
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}

impl<T: Dtype> core::ops::Deref for Cpu<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer.0
    }
}

impl<T: Dtype> Backend for Cpu<T> {
    type Dtype = T;
    type Buffer = CpuBuffer<T>;

    fn from(data: &[Self::Dtype]) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(data.to_vec().into())),
            shape: [data.len()].into(),
            stride: [1].into(),
        }
    }

    fn to_vec(&self) -> Vec<T> {
        let mut ret = Vec::with_capacity(self.shape.numel());
        for i in 0..self.shape.numel() {
            ret.push(self[self.row_i(i)]);
        }
        ret
    }

    fn empty(shape: &Shape) -> Self {
        Cpu {
            buffer: Arc::new(CpuBuffer(Vec::new())),
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

    fn rand(shape: &Shape) -> Self {
        let mut rng = crate::RNG.lock().unwrap();
        let mut out = vec![Default::default(); shape.numel()];
        for i in 0..shape.numel() {
            out[i] = rng.gen_range(0f64..1f64);
        }
        Cpu {
            buffer: Arc::new(CpuBuffer(
                out.iter().map(|o| T::from_f64(*o).unwrap()).collect(),
            )),
            shape: shape.clone(),
            stride: shape.strides(),
        }
    }

    fn add(&self, rhs: &Self) -> Self {
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Addd op: Did you forget to broadcast shape?"
        );
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
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Sub op: Did you forget to broadcast shape?"
        );
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
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Mul op: Did you forget to broadcast shape?"
        );
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
        assert!(
            (0..rhs.shape().numel()).any(|i| rhs[rhs.row_i(i)] != T::zero()),
            "Can not div self by zero"
        );
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Div op: Did you forget to broadcast shape? lhs:{} rhs: {}",
            self.shape,
            rhs.shape
        );
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
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Did you forgot to broadcast shape?"
        );
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
        assert!(
            self.shape.numel() == rhs.shape.numel(),
            "Did you forgot to broadcast shape?"
        );
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| {
                        if self.buffer.0[self.row_i(i)] < rhs.buffer.0[rhs.row_i(i)] {
                            T::one()
                        } else {
                            T::zero()
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
                    .map(|i| {
                        let n = self.buffer.0[self.row_i(i)].log2();
                        if n.is_nan() {
                            panic!("yo log2() return a NaN")
                        }
                        n
                    })
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
                    .map(|i| {
                        let n = self.buffer.0[self.row_i(i)].exp2();
                        if n.is_nan() {
                            panic!("yo exp2() return a NaN")
                        }
                        n
                    })
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
                    .map(|i| {
                        let n = self.buffer.0[self.row_i(i)].sin();
                        if n.is_nan() {
                            panic!("yo sin() return a NaN")
                        }
                        n
                    })
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
                    .map(|i| {
                        let n = self.buffer.0[self.row_i(i)].sqrt();
                        if n.is_nan() {
                            //panic!("yo sqrt() return a NaN")
                            T::zero()
                        } else {
                            n
                        }
                    })
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn sum(&self, _axis: Option<isize>, keepdim: bool) -> Self {
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
        let axis = if _axis < 0 {
            (self.shape.len() as isize + _axis) as usize
        } else {
            _axis as usize
        };
        let mut new_shape = self.shape().clone();
        new_shape.dims.remove(axis);
        // let mut out = Self::empty(&new_shape).const_like(T::zero());
        let mut out_inner_vec = vec![T::zero();new_shape.numel()];
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
                out_inner_vec[idx] += self.buffer.0[self.row_i(i)];
                idx += 1;
            } else {
                out_inner_vec[idx] += self.buffer.0[self.row_i(i)];
                if (i + 1) % numel_of_reduce_dim == 0 {
                    idx += 1;
                }
            }
        }
        let mut out = Self::empty(&new_shape);
        out.buffer = Arc::new(CpuBuffer(out_inner_vec.into()));
        if keepdim {
            let mut keepdim_shape = self.shape.clone();
            keepdim_shape[axis] = 1;
            out = out.reshape(keepdim_shape.clone());
        }
        out
    }

    fn max(&self, _axis: Option<isize>, keepdim: bool) -> Self {
        if _axis.is_none() {
            return Cpu {
                buffer: Arc::new(CpuBuffer(
                    vec![*self
                        .buffer
                        .0
                        .iter()
                        .reduce(|acc, e| if acc > e { acc } else { e })
                        .unwrap()]
                    .into(),
                )),
                shape: Shape::from([1]),
                stride: Shape::from([1]),
            };
        }
        let _axis = _axis.unwrap();
        let axis = if _axis < 0 {
            (self.shape.len() as isize + _axis) as usize
        } else {
            _axis as usize
        };
        let mut new_shape = self.shape().clone();
        new_shape.dims.remove(axis);
        // let mut out = Self::empty(&new_shape).const_like(T::min_value());
        let mut out_inner_vec = vec![T::min_value();new_shape.numel()];
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
                out_inner_vec[idx] = T::max(out_inner_vec[idx], self.buffer.0[self.row_i(i)]);
                idx += 1;
            } else {
                out_inner_vec[idx] = T::max(out_inner_vec[idx], self.buffer.0[self.row_i(i)]);
                if (i + 1) % numel_of_reduce_dim == 0 {
                    idx += 1;
                }
            }
        }
        let mut out = Self::empty(&new_shape);
        out.buffer = Arc::new(CpuBuffer(out_inner_vec.into()));
        if keepdim {
            let mut keepdim_shape = self.shape.clone();
            keepdim_shape[axis] = 1;
            out = out.reshape(keepdim_shape.clone());
        }
        out
    }

    fn _where(&self, x: &Self, y: &Self) -> Self {
        assert!(
            self.shape.numel() == x.shape.numel() || x.shape.numel() == y.shape.numel(),
            "where op: Did you forget to broadcast shape? self:{} x:{} y:{}",
            self.shape,
            x.shape,
            y.shape
        );
        Cpu {
            buffer: Arc::new(CpuBuffer(
                (0..self.shape.numel())
                    .map(|i| {
                        let ss = self.buffer.0[self.row_i(i)];
                        let xx = y.buffer.0[y.row_i(i)];
                        let yy = x.buffer.0[x.row_i(i)];
                        if ss == T::one() {
                            xx
                        } else {
                            yy
                        }
                    })
                    .collect(),
            )),
            shape: self.shape.clone(),
            stride: self.shape.strides(),
        }
    }

    fn permute<S: Into<Shape>>(&self, permute: S) -> Self {
        let mut out = self.clone();
        let permute = permute.into();
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
            // println!("reshape copied mem");
            out.buffer = Self::contiguous(&self).buffer.clone();
        }
        assert!(
            self.buffer.0.len() <= self.shape().numel(),
            "Invalid shape, cannot reshape into smaller self"
        );
        if self
            .shape
            .dims
            .iter()
            .filter(|&d| *d != 1)
            .into_iter()
            .eq(shape.dims.iter().filter(|&d| *d != 1))
        {
            let mut new_stride_dvec: VecDeque<usize> = VecDeque::new();
            new_stride_dvec.extend(
                self.shape
                    .dims
                    .iter()
                    .zip(self.stride.dims.iter())
                    .filter(|(sh, _)| **sh != 1)
                    .map(|(_, st)| *st),
            );

            let new_stride: Vec<usize> = shape
                .dims
                .iter()
                .map(|sh| {
                    if *sh == 1 {
                        0
                    } else {
                        new_stride_dvec.pop_front().unwrap()
                    }
                })
                .collect();
            out.stride = Shape::from(new_stride);
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
            .for_each(|(l, r)| {
                if l != r && *l != 1 {
                    panic!(
                        "Can not expand to shape. From: {} To: {}",
                        self.shape(),
                        shape
                    )
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
        let arg: Vec<(usize, usize)> = arg.into();
        assert!(arg.len() == self.shape.len() || arg.len() == 1);
        if arg.len() == 1 && self.shape.len() != 1 {
            arg.repeat(self.shape().len());
        }
        let new_shape = Shape::from(
            self.shape
                .dims
                .iter()
                .zip(arg.iter())
                .map(|(_, aarg)| aarg.0.abs_diff(aarg.1))
                .collect::<Vec<usize>>(),
        );
        //let mut out = Self::empty(&new_shape).const_like(T::zero());
        let mut out_inner_vec = vec![T::zero();new_shape.numel()];
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
                out_inner_vec[self_idx] = self[self.row_i(i)];
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
        let mut out = Self::empty(&new_shape);
        out.buffer = Arc::new(CpuBuffer(out_inner_vec.into()));
        out
    }

    fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: Self::Dtype) -> Self {
        let arg: Vec<(usize, usize)> = arg.into();
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
        //let mut out = Self::empty(&new_shape).const_like(const_value);
        let mut out_inner_vec = vec![const_value;new_shape.numel()];
        let mut iter_shape = vec![0usize; new_shape.len()];
        let mut self_idx = 0;

        for i in 0..new_shape.numel() {
            if iter_shape.iter().enumerate().all(|(ii, d)| {
                if *d >= arg[ii].0 && *d < arg[ii].0 + self.shape.dims[ii] {
                    true
                } else {
                    false
                }
            }) {
                // println!("{:?} {i}", iter_shape);
                out_inner_vec[i] = self[self.row_i(self_idx)];
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
        let mut out = Self::empty(&new_shape);
        out.buffer = Arc::new(CpuBuffer(out_inner_vec.into()));
        out
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn strides(&self) -> Shape {
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
}
