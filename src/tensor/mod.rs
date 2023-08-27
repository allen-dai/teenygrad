pub mod dtype;
pub mod shape;
pub mod core_ops;

use crate::backend::Backend;
use crate::prelude::*;
use rand_distr::StandardNormal;

pub struct Tensor<B: Backend> {
    pub(crate) inner: B,
    pub(crate) grad: Option<Vec<Tensor<B>>>,
}

impl<B: Backend> Tensor<B> {
    pub fn device(&self) -> String {
        std::any::type_name::<B>()
            .to_string()
            .split("::")
            .last()
            .unwrap()
            .to_string()
    }

    pub fn dtype(&self) -> String {
        std::any::type_name::<B::Dtype>().to_string()
    }

    pub fn from_vec<V: Into<Vec<B::Dtype>>, S: Into<Shape>>(data: V, shape: S) -> Self {
        let shape = shape.into();
        let data = data.into();
        assert!(data.len() == shape.numel());
        Self {
            inner: B::from_vec(data, &shape),
            grad: None,
        }
    }

    pub fn randn<S: Into<Shape>>(shape: S) -> Self
    where
        StandardNormal: rand_distr::Distribution<B::Dtype>,
    {
        Self {
            inner: B::rand(&shape.into(), StandardNormal),
            grad: None,
        }
    }

    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        use num_traits::Zero;
        Self {
            inner: B::empty(&shape.into()).const_like(B::Dtype::zero()),
            grad: None,
        }
    }

    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        use num_traits::One;
        Self {
            inner: B::empty(&shape.into()).const_like(B::Dtype::one()),
            grad: None,
        }
    }

    pub fn sum(&self, axis: isize) -> Self {
        Self {
            inner: B::sum(&self.inner, Some(axis)),
            grad: None,
        }
    }

    pub fn sum_all(&self) -> Self {
        Self {
            inner: B::sum(&self.inner, None),
            grad: None,
        }
    }

    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::reshape(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn permute<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::permute(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::expand(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn to_vec(&self) -> Vec<B::Dtype> {
        self.inner.to_vec()
    }

    pub fn shape(&self) -> Shape {
        B::shape(&self.inner)
    }

    pub fn stride(&self) -> Shape {
        B::stride(&self.inner)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        let n1 = self.shape().len();
        let n2 = rhs.shape().len();
        let mut a_shape = self.shape();
        let last = a_shape.dims.pop().unwrap();
        a_shape
            .dims
            .extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        a_shape.dims.push(last);
        let mut a = self.reshape(a_shape);
        let mut b_shape = rhs.shape();
        b_shape.dims = b_shape.dims[0..b_shape.len() - 2].to_vec();
        b_shape
            .dims
            .extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        b_shape
            .dims
            .extend_from_slice(&rhs.shape().dims[rhs.shape().len() - n2.min(2)..]);
        let mut order = (0..b_shape.len()).collect::<Vec<usize>>();
        order.swap(b_shape.len() - 1, b_shape.len() - 2.min(n2));
        let mut b = rhs.reshape(b_shape).permute(order);
        // println!("{:?}", b);
        // println!("{:?} {:?}\n{:?} {:?}", a.shape(), a.stride(), b.shape(), b.stride());
        let (a, b) = Self::_broadcast(a, b);
        //println!("{:?} {:?}\n{:?} {:?}", a.shape(), a.stride(), b.shape(), b.stride());
        (a * b).sum(-1)
    }

    pub fn _broadcast(mut x: Self, mut y: Self) -> (Self, Self) {
        let mut xshape = x.shape();
        let mut yshape = y.shape();
        if xshape == yshape {
            return (x, y);
        }
        let shape_delta = xshape.len() - yshape.len();
        if shape_delta > 0 {
            let mut ysh = vec![1; shape_delta];
            ysh.extend_from_slice(&yshape.dims);
            y = y.reshape(ysh);
        }
        yshape = y.shape();
        if xshape == yshape {
            return (x, y);
        }

        let shape_ret = Shape::from(
            xshape
                .dims
                .iter()
                .zip(yshape.dims.iter())
                .map(|(x, y)| *x.max(y))
                .collect::<Vec<usize>>(),
        );
        if xshape != shape_ret {
            x = x.expand(shape_ret.clone());
        }
        if yshape != shape_ret {
            y = y.expand(shape_ret.clone());
        }
        (x, y)
    }

    pub fn t(&self, d1: isize, d2: isize) -> Self {
        let d1 = if d1 < 0 {
            (self.shape().len() as isize + d1) as usize
        } else {
            d1 as usize
        };
        let d2 = if d2 < 0 {
            (self.shape().len() as isize + d2) as usize
        } else {
            d2 as usize
        };

        let mut new_shape = (0..self.shape().len()).collect::<Vec<usize>>();
        new_shape.swap(d1, d2);
        self.permute(new_shape)
    }

    pub fn shrink<A: Into<Vec<(usize, usize)>>>(&self, arg: A) -> Self {
        Tensor {
            inner: self.inner.shrink(arg),
            grad: None,
        }
    }

    pub fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: B::Dtype) -> Self {
        Tensor {
            inner: self.inner.pad(arg, const_value),
            grad: None,
        }
    }

    #[rustfmt::skip]
    pub fn _pool<S: Into<Shape>>(&self, k_: S, stride: S, dilation: usize) -> Self {
        let self_shape = self.shape();
        let (k_, s_) = (k_.into(), stride.into());
        assert!(self_shape.len() >= k_.len(), "can't pool {self_shape:?} with {k_:?}");
        let d_ = Shape::from(vec![dilation;k_.len()]);
        assert!(k_.len() == s_.len() && s_.len() == d_.len(), "stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}");
        let slc_prefix: Vec<(usize, usize)> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| (0, *sh))
            .collect();
        let prefix: Vec<usize> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| *sh)
            .collect();
        let i_: Vec<usize> = self_shape.dims[self_shape.len() - k_.len()..]
            .iter()
            .map(|sh| *sh)
            .collect();

        let mut xup = if k_.dims.iter().zip(s_.dims.iter()).any(|(k, s)| k > s) || d_.dims.iter().any(|d| *d != 1)
        {
            let mut xup;
            let o_: Vec<usize> = i_
                .iter()
                .zip(d_.dims.iter().zip(k_.dims.iter().zip(s_.dims.iter())))
                .map(|(i, (d, (k, s)))| (i - d * (k - 1) - 1) / s + 1)
                .collect();
            let e_: Vec<usize> = k_
                .dims
                .iter()
                .zip(i_.iter().zip(d_.dims.iter()))
                .map(|(k, (i, d))| k * (i + d) / i)
                .collect();
            let mut tmp = prefix.clone();
            tmp.extend(i_.iter().map(|i| vec![1, *i]).collect::<Vec<Vec<usize>>>().iter().flatten());
            let mut tmp2 = prefix.clone();
            tmp2.extend(e_.iter().zip(i_.iter()).map(|(e, i)| vec![*e, *i]).collect::<Vec<Vec<usize>>>().iter().flatten());
            let mut tmp3 = prefix.clone();
            tmp3.extend(e_.iter().zip(i_.iter()).map(|(e, i)| *e * *i));
            xup = self.reshape(tmp).expand(tmp2).reshape(tmp3);

            // Stride by dilation
            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(i_.iter().zip(d_.dims.iter())).map(|(k, (i, d))|{
                (0, k*(i+d))
            }));
            xup = xup.slice(tmp, B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(i_.iter().zip(d_.dims.iter())).map(|(k, (i, d))| vec![*k, *i + *d]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter().zip(s_.dims.iter())).map(|(k, (o, s))| vec![(0, *k),(0, *o * *s)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = xup.slice(tmp, B::Dtype::zero());

            // handle stride, and permute to move reduce to the end
            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter().zip(s_.dims.iter())).map(|(k, (o, s))| vec![*k, *o, *s]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter()).map(|(k, o)| vec![(0, *k), (0, *o), (0, 1)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = xup.slice(tmp, B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter()).map(|(k, o)| vec![*k, *o]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = vec![0usize; prefix.len()];
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2 + 1));
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2));
            xup.permute(tmp)
        } else {
            let mut xup;

            let o_:Vec<usize> = i_.iter().zip(s_.dims.iter().zip(k_.dims.iter())).map(|(i, (s, k))|
                (*i+(*s-*k))*s
            ).collect();

            let mut tmp = slc_prefix.clone();
            tmp.extend(o_.iter().zip(s_.dims.iter()).map(|(o, s)| vec![(0, *o * *s)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = self.slice(tmp, B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(o_.iter().zip(s_.dims.iter()).map(|(o, s)| vec![*o, *s]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(o_.iter().zip(k_.dims.iter()).map(|(o, k)| vec![(0, *o), (0, *k)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = self.slice(tmp, B::Dtype::zero());

            let mut tmp = vec![0usize; prefix.len()];
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2));
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2 + 1));
            xup.permute(tmp)
        };
        xup
    }

    #[rustfmt::skip]
    pub fn slice<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: B::Dtype) -> Self {
        let arg = arg.into();
        let self_shape = self.shape();
        let padding: Vec<(usize, usize)> = arg.iter().enumerate().map(|(i, p)|
            (0, 0.max(if p.1 >= self_shape[i] { p.1 - self_shape[i] } else { 0 }))
        ).collect();
        let shrink: Vec<(usize, usize)> = arg.iter().enumerate().map(|(i, p)|
            (p.0 + padding[i].0, p.1 + padding[i].0)
        ).collect();
        self.pad(padding, const_value).shrink(shrink)
    }
}

#[test]
fn sum_axis() {
    use crate::prelude::*;
    let n = 2 * 3;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..n + 1)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [2, 3],
    );
    let y = t.sum(1);
    assert!(!vec![6.0f32, 15.0f32]
        .iter()
        .zip(y.to_vec())
        .any(|(l, r)| *l != r));

    let n = 4 * 2 * 3 * 3;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..n + 1)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [4, 2, 3, 3],
    );

    let y = t.sum(0);
    assert!(!vec![
        112., 116., 120., 124., 128., 132., 136., 140., 144., 148., 152., 156., 160., 164., 168.,
        172., 176., 180.
    ]
    .iter()
    .zip(y.to_vec())
    .any(|(l, r)| *l != r));

    let y = t.sum(1);
    assert!(!vec![
        11., 13., 15., 17., 19., 21., 23., 25., 27., 47., 49., 51., 53., 55., 57., 59., 61., 63.,
        83., 85., 87., 89., 91., 93., 95., 97., 99., 119., 121., 123., 125., 127., 129., 131.,
        133., 135.
    ]
    .iter()
    .zip(y.to_vec())
    .any(|(l, r)| *l != r));

    let y = t.sum(2);
    assert!(!vec![
        12., 15., 18., 39., 42., 45., 66., 69., 72., 93., 96., 99., 120., 123., 126., 147., 150.,
        153., 174., 177., 180., 201., 204., 207.
    ]
    .iter()
    .zip(y.to_vec())
    .any(|(l, r)| *l != r));

    let y = t.sum(3);
    assert!(!vec![
        6., 15., 24., 33., 42., 51., 60., 69., 78., 87., 96., 105., 114., 123., 132., 141., 150.,
        159., 168., 177., 186., 195., 204., 213.
    ]
    .iter()
    .zip(y.to_vec())
    .any(|(l, r)| *l != r));
}

#[test]
fn matmul() {
    let a = Tensor::<Cpu>::from_vec([1., 2., 3., 4., 5., 6.], [2, 3]);
    let b = Tensor::<Cpu>::from_vec([10., 11., 20., 21., 30., 31.], [3, 2]);
    &a + 1.0;
    let y = a.matmul(&b);
    assert!(!vec![140., 146., 320., 335.]
        .iter()
        .zip(y.to_vec())
        .any(|(l, r)| *l != r));

    let a = Tensor::<Cpu>::from_vec(
        (1..=4 * 9)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [4, 9],
    );
    let b = Tensor::<Cpu>::from_vec(
        (1..=9 * 2)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [9, 2],
    );
    let y = a.matmul(&b);
    assert!(!vec![525., 570., 1254., 1380., 1983., 2190., 2712., 3000.]
        .iter()
        .zip(y.to_vec())
        .any(|(l, r)| *l != r));
}

#[test]
fn pool() {
    let n = 9;
    let mut a = Tensor::<Cpu>::from_vec(
        (1..=n * n * n)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [n, n, n],
    );
    let k = Shape::from([3, 3]);
    let stride = Shape::from(vec![1; k.len()]);
    let t = a._pool(k, stride, 1);
}
