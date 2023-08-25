pub mod dtype;
pub mod shape;
use rand_distr::StandardNormal;

use crate::backend::Backend;
use crate::prelude::*;

pub struct Tensor<B: Backend> {
    pub(crate) inner: B,
    pub(crate) grad: Option<Vec<Tensor<B>>>,
}

impl<B: Backend> core::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{:?}Shape:{:?} Stride:{:?} Dtype:{} Device:{}\n",
            self.inner,
            self.inner.shape(),
            self.inner.stride(),
            self.dtype(),
            self.device(),
        )
    }
}

impl<B: Backend> core::ops::Add for Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::add(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend> core::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::sub(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend> core::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::mul(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend> core::ops::Div for Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::div(&self.inner, &rhs.inner),
            grad: None,
        }
    }
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

    pub fn reshape<S: Into<Shape>>(&mut self, shape: S) -> Self {
        Self {
            inner: B::reshape(&mut self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn permute<S: Into<Shape>>(&mut self, shape: S) -> Self {
        Self {
            inner: B::permute(&mut self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn expand<S: Into<Shape>>(&mut self, shape: S) -> Self {
        Self {
            inner: B::expand(&mut self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn to_vec(&self) -> Vec<B::Dtype> {
        self.inner.to_vec()
    }

    pub fn shape(&self) -> Shape {
        B::shape(&self.inner)
    }
}

#[test]
fn sum_axis() {
    use crate::prelude::*;
    use num_traits::FromPrimitive;
    let n = 2 * 3;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..n + 1)
            .into_iter()
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
            .into_iter()
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
