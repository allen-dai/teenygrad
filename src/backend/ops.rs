use super::Backend;
use crate::{
    prelude::Tensor,
    tensor::{dtype::Dtype, shape::Shape},
};
use num_traits::FromPrimitive;
use std::marker::PhantomData;

pub trait Function<B: Backend> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B;
    fn backward(&mut self, grad: B) -> Grad<B>;
}

#[derive(Debug)]
pub enum Grad<B: Backend> {
    Contiguous(B),
    Sin(B),
    Log(B),
    Exp(B),
    Sqrt(B),
    Max(B),
    Sum(B),
    Add(Option<B>, Option<B>),
    Sub(Option<B>, Option<B>),
    Mul(Option<B>, Option<B>),
    Div(Option<B>, Option<B>),
    Sigmoid(B),
    Relu(B),
}

// impl<B: Backend> core::fmt::Debug for Grad<B> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match &self {
//             Grad::Contiguous(x)
//             | Grad::Sin(x)
//             | Grad::Log(x)
//             | Grad::Exp(x)
//             | Grad::Sqrt(x)
//             | Grad::Max(x)
//             | Grad::Sum(x)
//             | Grad::Sigmoid(x)
//             | Grad::Relu(x) => write!(f, "{x:?}"),
//             Grad::Add(x, y) | Grad::Sub(x, y) | Grad::Mul(x, y) | Grad::Div(x, y) => {
//                 write!(f, "x:{x:?}\ny:{y:?}")
//             }
//         }
//     }
// }

macro_rules! df32 {
    ($t: expr) => {
        B::Dtype::from_f32($t).unwrap()
    };
}

pub struct Contiguous<B: Backend> {
    phantom: PhantomData<B>,
}

impl<B: Backend> Function<B> for Contiguous<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        x.contiguous()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Contiguous(grad)
    }
}

pub struct Sin<B: Backend> {
    pub(crate) x: B,
}

impl<B: Backend> Function<B> for Sin<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.x = x;
        self.x.sin()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Sin(
            self.x
                .const_like(B::Dtype::PI / B::Dtype::from_f32(2.0).unwrap())
                .sub(&self.x)
                .sin()
                .mul(&grad),
        )
    }
}

pub struct Log<B: Backend> {
    pub(crate) x: B,
}

impl<B: Backend> Function<B> for Log<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.x = x;
        self.x
            .log2()
            .mul(&self.x.const_like(df32!(2.0f32.log(f32::EPSILON))))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Log(grad.div(&self.x))
    }
}

pub struct Exp<B: Backend> {
    pub(crate) ret: B,
}

impl<B: Backend> Function<B> for Exp<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.ret = x
            .mul(&x.const_like(df32!(1f32 / 2.0f32.log(f32::EPSILON))))
            .exp2();
        self.ret.clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Exp(self.ret.mul(&grad))
    }
}

pub struct Sqrt<B: Backend> {
    pub(crate) ret: Option<B>,
}

impl<B: Backend> Function<B> for Sqrt<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.ret = Some(x.sqrt());
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let ret = self.ret.as_ref().unwrap();
        Grad::Sqrt(grad.div(&ret.mul(&ret.const_like(df32!(0.0f32)))))
    }
}

pub struct Sum<B: Backend> {
    pub(crate) input_shape: Shape,
    pub(crate) phantom: PhantomData<B>,
}

impl<B: Backend> Function<B> for Sum<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.input_shape = shape.unwrap().into();
        let sum_axis = x
            .shape()
            .dims
            .iter()
            .zip(self.input_shape.dims.iter())
            .position(|(&x, &sh)| x != sh)
            .unwrap();
        x.sum(Some(sum_axis as isize))
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        grad.expand(self.input_shape.clone());
        Grad::Sum(grad)
    }
}

pub struct Max<B: Backend> {
    pub(crate) x: B,
    pub(crate) ret: B,
}

impl<B: Backend> Function<B> for Max<B> {
    // TODO: we might need to pass shape into max()
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.ret = x.max();
        self.x = x;
        self.ret.clone()
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        let max_is_1s = self
            .x
            .const_like(df32!(1.0))
            .sub(&self.x.cmplt(&self.ret.expand(self.x.shape())));
        let sum_axis = max_is_1s
            .shape()
            .dims
            .iter()
            .zip(grad.shape().dims.iter())
            .position(|(&x, &sh)| x != sh)
            .unwrap();
        let div = max_is_1s.sum(Some(sum_axis as isize)).expand(self.x.shape());
        Grad::Max(max_is_1s.div(&div).mul(&grad.expand(self.x.shape())))
    }
}

pub struct Less;

impl<B: Backend> Function<B> for Less {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        x.cmplt(&y.expect("Less fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        unreachable!("Less op can not bwd")
    }
}

pub struct Add {
    need_input_grad: [bool; 2],
}

impl<B: Backend> Function<B> for Add {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        x.add(&y.expect("Add fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(grad.clone())
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(grad.clone())
        } else {
            None
        };
        Grad::Add(x, y)
    }
}

pub struct Sub {
    need_input_grad: [bool; 2],
}

impl<B: Backend> Function<B> for Sub {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        x.sub(&y.expect("Sub fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(grad.const_like(df32!(0.0)).sub(&grad))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(grad.const_like(df32!(0.0)).sub(&grad))
        } else {
            None
        };
        Grad::Sub(x, y)
    }
}

pub struct Mul<B: Backend> {
    need_input_grad: [bool; 2],
    x: B,
    y: B,
}

impl<B: Backend> Function<B> for Mul<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.x = x;
        self.y = y.expect("Mul fwd op expects rhs");
        self.x.mul(&self.y)
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(self.y.mul(&grad))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(self.x.mul(&grad))
        } else {
            None
        };
        Grad::Mul(x, y)
    }
}

pub struct Div<B: Backend> {
    need_input_grad: [bool; 2],
    x: B,
    y: B,
}

impl<B: Backend> Function<B> for Div<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.x = x;
        self.y = y.expect("Mul fwd op expects rhs");
        self.x.div(&self.y)
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(grad.div(&self.y))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(
                grad.const_like(df32!(0.0))
                    .sub(&grad)
                    .mul(&self.x)
                    .div(&self.y.mul(&self.y)),
            )
        } else {
            None
        };
        Grad::Div(x, y)
    }
}

pub struct Sigmoid<B: Backend> {
    pub(crate) ret: B,
}

impl<B: Backend> Function<B> for Sigmoid<B> {
    fn forward(&mut self, x: B, shape: Option<Shape>, y: Option<B>) -> B {
        self.ret = x.const_like(df32!(1.0)).div(
            &x.const_like(df32!(1.0)).add(
                &x.mul(&x.const_like(df32!(-1.0 / 2.0f32.log(f32::EPSILON))))
                    .exp2(),
            ),
        );
        self.ret.clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Sigmoid(
            self.ret
                .mul(&self.ret.const_like(df32!(1.0)).sub(&self.ret))
                .mul(&grad),
        )
    }
}

pub struct Relu<B: Backend> {
    pub(crate) ret: Option<B>,
}

impl<B: Backend> Function<B> for Relu<B> {
    fn forward(&mut self, x: B, _: Option<Shape>, y: Option<B>) -> B {
        self.ret = Some(x.bmax(&x.const_like(df32!(0.0))));
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::Relu(
            self.ret
                .as_ref()
                .unwrap()
                .const_like(B::Dtype::from_f32(0.0).unwrap())
                .cmplt(&self.ret.as_ref().unwrap())
                .mul(&grad),
        )
    }
}
