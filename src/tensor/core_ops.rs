use crate::backend::ops::*;
use crate::prelude::*;

use super::id::tensor_id;

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
        Tensor::_add(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Add for &Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::_add(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Add<F> for Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_add(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Add<F> for &Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_add(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::_sub(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Sub for &Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::_sub(self, rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Sub<F> for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_sub(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Sub<F> for &Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_sub(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::_mul(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Mul for &Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::_mul(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Mul<F> for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_mul(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Mul<F> for &Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_mul(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Div for Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::_div(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Div for &Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::_div(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Div<F> for Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_div(&self, &rhs)
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Div<F> for &Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: F) -> Self::Output {
        let rhs = Tensor::from_vec(
            [B::Dtype::from_f64(rhs.to_f64().unwrap()).unwrap()],
            Shape::from([1]),
        );
        Tensor::_div(&self, &rhs)
    }
}

impl<B: Backend> core::ops::Neg for Tensor<B> {
    type Output = Tensor<B>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<B: Backend> core::ops::Neg for &Tensor<B> {
    type Output = Tensor<B>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}
