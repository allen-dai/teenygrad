use crate::{
    prelude::*,
    tensor::id::{tensor_id, TensorId},
};
use dyn_clone::DynClone;
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

pub fn argsort<V: Into<Vec<usize>>>(shape: V) -> Vec<usize> {
    let shape = shape.into();
    let mut out = (0..shape.len()).into_iter().collect::<Vec<_>>();
    out.sort_by_key(|&i| &shape[i]);
    out
}

#[derive(Debug, Clone)]
pub struct Ctx<B: Backend>(pub(crate) HashMap<TensorId, Tensor<B>>);

impl<B: Backend> Default for Ctx<B> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<B: Backend> core::ops::Deref for Ctx<B> {
    type Target = HashMap<TensorId, Tensor<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> core::ops::DerefMut for Ctx<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait Function<B: Backend>: DynClone + core::fmt::Debug {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B;
    fn backward(&mut self, grad: B) -> Grad<B>;
    fn ctx_mut(&mut self) -> &mut Ctx<B>;
    fn ctx(&self) -> &Ctx<B>;
    fn apply(&mut self, x: &Tensor<B>, shape: Option<Shape>, y: Option<B>) -> Tensor<B>
    where
        Self: 'static + Sized,
    {
        let inner = self.forward(&x.inner, shape, y.as_ref());
        let require_grad = x.require_grad;
        if require_grad {
            self.ctx_mut().insert(x.id, x.clone());
        }
        Tensor {
            inner,
            require_grad,
            _ctx: if require_grad {
                Some(dyn_clone::clone_box(&*self))
            } else {
                None
            },
            id: x.id,
            grad: None,
        }
    }
}

dyn_clone::clone_trait_object!(<B> Function<B> where B: Backend);

#[derive(Debug, Clone)]
pub enum Grad<B: Backend> {
    One(B),
    Two(Option<B>, Option<B>),
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

#[derive(Clone, Debug)]
pub struct Contiguous<B: Backend> {
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Contiguous<B> {
    fn default() -> Self {
        Self {
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Contiguous<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        x.contiguous()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(grad)
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sin<B: Backend> {
    pub(crate) x: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Sin<B> {
    fn default() -> Self {
        Self {
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Sin<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.x = Some(x.clone());
        x.sin()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = self.x.as_ref().unwrap();
        Grad::One(
            x.const_like(B::Dtype::PI / B::Dtype::from_f32(2.0).unwrap())
                .sub(x)
                .sin()
                .mul(&grad),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Log<B: Backend> {
    pub(crate) x: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Log<B> {
    fn default() -> Self {
        Self {
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Log<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.x = Some(x.clone());
        x.log2().mul(&x.const_like(df32!(2.0f32.log(f32::EPSILON))))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(grad.div(self.x.as_ref().unwrap()))
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Exp<B: Backend> {
    pub(crate) ret: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Exp<B> {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Exp<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        let ret = x
            .mul(&x.const_like(df32!(1f32 / 2.0f32.log(f32::EPSILON))))
            .exp2();
        self.ret = Some(ret.clone());
        ret
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(self.ret.as_ref().unwrap().mul(&grad))
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sqrt<B: Backend> {
    pub(crate) ret: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Sqrt<B> {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Sqrt<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.ret = Some(x.sqrt());
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let ret = self.ret.as_ref().unwrap();
        Grad::One(grad.div(&ret.mul(&ret.const_like(df32!(0.0f32)))))
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sum<B: Backend> {
    pub(crate) input_shape: Option<Shape>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Sum<B> {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Sum<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        let mut shape = shape.unwrap();
        self.input_shape = Some(x.shape());
        if shape.len() == 1 {
            return x.sum(None, false);
        }
        let (keepdim, axis) = if shape.len() - x.shape().len() == 1 {
            //TODO: hack, need change
            (true, *shape.dims.last().unwrap())
        } else {
            (false, shape.dims.iter().position(|e| *e == 0).unwrap())
        };
        x.sum(Some(axis as isize), keepdim)
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        Grad::One(
            grad.expand(
                self.input_shape
                    .as_ref()
                    .expect("Sum bwd should have a input_shape")
                    .clone(),
            ),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Max<B: Backend> {
    pub(crate) x: Option<B>,
    pub(crate) ret: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Max<B> {
    fn default() -> Self {
        Self {
            x: None,
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Max<B> {
    // TODO: we might need to pass shape into max()
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.ret = Some(x.max());
        self.x = Some(x.clone());
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        let x_ref = self.x.as_ref().unwrap();
        let ret_ref = self.ret.as_ref().unwrap();
        let max_is_1s = x_ref
            .const_like(df32!(1.0))
            .sub(&x_ref.cmplt(&ret_ref.expand(x_ref.shape())));
        let sum_axis = max_is_1s
            .shape()
            .dims
            .iter()
            .zip(grad.shape().dims.iter())
            .position(|(&x, &sh)| x != sh)
            .unwrap();
        let div = max_is_1s
            .sum(Some(sum_axis as isize), false)
            .expand(x_ref.shape());
        Grad::One(max_is_1s.div(&div).mul(&grad.expand(x_ref.shape())))
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Less<B: Backend> {
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Less<B> {
    fn default() -> Self {
        Self {
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Less<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        x.cmplt(&y.expect("Less fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        unreachable!("Less op can not bwd")
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Add<B: Backend> {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<B>,
    pub(crate) y: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Add<B> {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Add<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
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
        Grad::Two(x, y)
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sub<B: Backend> {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<B>,
    pub(crate) y: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Sub<B> {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Sub<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
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
        Grad::Two(x, y)
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Mul<B: Backend> {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<B>,
    pub(crate) y: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Mul<B> {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Mul<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.x = Some(x.clone());
        self.y = Some(y.expect("Mul fwd op expects rhs").clone());
        x.mul(y.as_ref().unwrap())
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(self.y.as_ref().unwrap().mul(&grad))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            Some(self.x.as_ref().unwrap().mul(&grad))
        } else {
            None
        };
        Grad::Two(x, y)
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Div<B: Backend> {
    pub(crate) need_input_grad: [bool; 2],
    pub(crate) x: Option<B>,
    pub(crate) y: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Div<B> {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false],
            x: None,
            y: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Div<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.x = Some(x.clone());
        self.y = Some(y.expect("Mul fwd op expects rhs").clone());
        x.div(y.as_ref().unwrap())
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(grad.div(self.y.as_ref().unwrap()))
        } else {
            None
        };
        let y = if self.need_input_grad[1] {
            let y_ref = self.y.as_ref().unwrap();
            Some(
                grad.const_like(df32!(0.0))
                    .sub(&grad)
                    .mul(self.x.as_ref().unwrap())
                    .div(&y_ref.mul(y_ref)),
            )
        } else {
            None
        };
        Grad::Two(x, y)
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Sigmoid<B: Backend> {
    pub(crate) ret: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Sigmoid<B> {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Sigmoid<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.ret = Some(
            x.const_like(df32!(1.0)).div(
                &x.const_like(df32!(1.0)).add(
                    &x.mul(&x.const_like(df32!(-1.0 / 2.0f32.log(f32::EPSILON))))
                        .exp2(),
                ),
            ),
        );
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let ret_ref = self.ret.as_ref().unwrap();
        Grad::One(
            ret_ref
                .mul(
                    &ret_ref
                        .const_like(df32!(1.0))
                        .sub(&self.ret.as_ref().unwrap()),
                )
                .mul(&grad),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Relu<B: Backend> {
    pub(crate) ret: Option<B>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Relu<B> {
    fn default() -> Self {
        Self {
            ret: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Relu<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.ret = Some(x.bmax(&x.const_like(df32!(0.0))));
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(
            self.ret
                .as_ref()
                .unwrap()
                .const_like(B::Dtype::from_f32(0.0).unwrap())
                .cmplt(&self.ret.as_ref().unwrap())
                .mul(&grad),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Reshape<B: Backend> {
    pub(crate) input_shape: Option<Shape>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Reshape<B> {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Reshape<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.input_shape = Some(x.shape());
        x.reshape(shape.expect("Reshape mlops expect a shape"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(
            grad.reshape(
                self.input_shape
                    .as_ref()
                    .expect("Reshape backward should already have a shape")
                    .clone(),
            ),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Permute<B: Backend> {
    pub(crate) permute_order: Option<Shape>,
    pub(crate) ctx: Ctx<B>,
    phantom: PhantomData<B>,
}

impl<B: Backend> Default for Permute<B> {
    fn default() -> Self {
        Self {
            permute_order: None,
            phantom: PhantomData,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Permute<B> {
    fn forward(&mut self, x: &B, shape: Option<Shape>, y: Option<&B>) -> B {
        self.permute_order = Some(x.shape());
        x.permute(shape.expect("Permute mlops expect a permute order"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(
            grad.permute(argsort(
                &*self
                    .permute_order
                    .as_ref()
                    .expect("Permute bwd order should not be empty")
                    .dims,
            )),
        )
    }

    fn ctx_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn ctx(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[test]
fn test_ctx() {
    let mut o = Tensor::<Cpu>::randn([3, 3]);
    o.require_grad = true;
    let mut t = ((&o + 100.0) * -24.0).sum_all();
    t.backward();
    println!("{:?}", t.grad);
    // TODO: this is just checking backward, need better test
}
