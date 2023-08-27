use crate::prelude::*;

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

impl<B: Backend> core::ops::Add for &Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::add(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Add<F> for &Tensor<B>
where
    Vec<B::Dtype>: From<[F; 1]>,
{
    type Output = Tensor<B>;

    fn add(self, rhs: F) -> Self::Output {
        Tensor {
            inner: B::add(&self.inner, &Tensor::from_vec([rhs], Shape::from([1])).inner),
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

impl<B: Backend> core::ops::Sub for &Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::sub(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Sub<F> for &Tensor<B>
where
    Vec<B::Dtype>: From<[F; 1]>,
{
    type Output = Tensor<B>;

    fn sub(self, rhs: F) -> Self::Output {
        Tensor {
            inner: B::sub(&self.inner, &Tensor::from_vec([rhs], Shape::from([1])).inner),
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

impl<B: Backend> core::ops::Mul for &Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::mul(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Mul<F> for &Tensor<B>
where
    Vec<B::Dtype>: From<[F; 1]>,
{
    type Output = Tensor<B>;

    fn mul(self, rhs: F) -> Self::Output {
        Tensor {
            inner: B::mul(&self.inner, &Tensor::from_vec([rhs], Shape::from([1])).inner),
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


impl<B: Backend> core::ops::Div for &Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor {
            inner: B::div(&self.inner, &rhs.inner),
            grad: None,
        }
    }
}

impl<B: Backend, F: num_traits::Float> core::ops::Div<F> for &Tensor<B>
where
    Vec<B::Dtype>: From<[F; 1]>,
{
    type Output = Tensor<B>;

    fn div(self, rhs: F) -> Self::Output {
        Tensor {
            inner: B::mul(&self.inner, &Tensor::from_vec([rhs], Shape::from([1])).inner),
            grad: None,
        }
    }
}
