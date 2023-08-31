use crate::prelude::*;

impl<B: Backend> core::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{:?}Shape:{:?} Stride:{:?} Dtype:{} Device:{} Id:{:?} require_grad:{} grad:{:?}\nctx:{:?}\n",
            self.inner,
            self.inner.shape(),
            self.inner.strides(),
            self.dtype(),
            self.device(),
            self.id.0,
            self.require_grad,
            self.grad,
            self._ctx,
        )
    }
}

impl<B: Backend> core::fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{:?}Shape:{:?} Stride:{:?} Dtype:{} Device:{}\n",
            self.inner,
            self.inner.shape(),
            self.inner.strides(),
            self.dtype(),
            self.device(),
        )
    }
}

macro_rules! core_impl {
    ($op:tt, $fn:tt) => {
        impl<B: Backend> core::ops::$op for Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<&Tensor<B>> for Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: &Tensor<B>) -> Self::Output {
                Tensor::$fn(&self, rhs)
            }
        }

        impl<B: Backend> core::ops::$op for &Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Tensor::$fn(self, rhs)
            }
        }
    };
}

core_impl!(Add, add);
core_impl!(Sub, sub);
core_impl!(Mul, mul);
core_impl!(Div, div);

macro_rules! core_impl_num {
    ($op:tt, $fn:tt, $t:ty, $from:ident) => {

        impl<B: Backend> core::ops::$op<$t> for Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::from([B::Dtype::$from(rhs).unwrap()]);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<$t> for &Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: $t) -> Self::Output {
                let rhs = Tensor::from([B::Dtype::$from(rhs).unwrap()]);
                Tensor::$fn(self, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<&$t> for Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::from([B::Dtype::$from(*rhs).unwrap()]);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<&$t> for &Tensor<B> {
            type Output = Tensor<B>;
            fn $fn(self, rhs: &$t) -> Self::Output {
                let rhs = Tensor::from([B::Dtype::$from(*rhs).unwrap()]);
                Tensor::$fn(&self, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<Tensor<B>> for $t {
            type Output = Tensor<B>;
            fn $fn(self, rhs: Tensor<B>) -> Self::Output {
                let lhs = Tensor::from([B::Dtype::$from(self).unwrap()]);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<Tensor<B>> for &$t {
            type Output = Tensor<B>;
            fn $fn(self, rhs: Tensor<B>) -> Self::Output {
                let lhs = Tensor::from([B::Dtype::$from(*self).unwrap()]);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<&Tensor<B>> for $t {
            type Output = Tensor<B>;
            fn $fn(self, rhs: &Tensor<B>) -> Self::Output {
                let lhs = Tensor::from([B::Dtype::$from(self).unwrap()]);
                Tensor::$fn(&lhs, &rhs)
            }
        }

        impl<B: Backend> core::ops::$op<&Tensor<B>> for &$t {
            type Output = Tensor<B>;
            fn $fn(self, rhs: &Tensor<B>) -> Self::Output {
                let lhs = Tensor::from([B::Dtype::$from(*self).unwrap()]);
                Tensor::$fn(&lhs, rhs)
            }
        }
    };
}

core_impl_num!(Add, add, f32, from_f32);
core_impl_num!(Sub, sub, f32, from_f32);
core_impl_num!(Mul, mul, f32, from_f32);
core_impl_num!(Div, div, f32, from_f32);

core_impl_num!(Add, add, f64, from_f64);
core_impl_num!(Sub, sub, f64, from_f64);
core_impl_num!(Mul, mul, f64, from_f64);
core_impl_num!(Div, div, f64, from_f64);

core_impl_num!(Add, add, i32, from_i32);
core_impl_num!(Sub, sub, i32, from_i32);
core_impl_num!(Mul, mul, i32, from_i32);
core_impl_num!(Div, div, i32, from_i32);

core_impl_num!(Add, add, usize, from_usize);
core_impl_num!(Sub, sub, usize, from_usize);
core_impl_num!(Mul, mul, usize, from_usize);
core_impl_num!(Div, div, usize, from_usize);

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
