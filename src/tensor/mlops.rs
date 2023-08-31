use crate::{
    prelude::*,
    tensor::id::{tensor_id, TensorId},
};
use dyn_clone::DynClone;
use std::marker::PhantomData;

pub fn argsort<V: Into<Vec<usize>>>(shape: V) -> Vec<usize> {
    let shape = shape.into();
    let mut out = (0..shape.len()).into_iter().collect::<Vec<_>>();
    out.sort_by_key(|&i| &shape[i]);
    out
}

pub fn shape_to_axis(old_shape: Shape, new_shape: Shape) -> Vec<usize> {
    assert!(old_shape.len() == new_shape.len());
    let mut ret = Vec::new();
    for (i, (o, d)) in old_shape.dims.iter().zip(new_shape.dims.iter()).enumerate() {
        if o != d {
            ret.push(i)
        }
    }
    ret
}

#[derive(Debug, Clone)]
pub struct Ctx<B: Backend>(pub(crate) Vec<Tensor<B>>);

impl<B: Backend> Default for Ctx<B> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<B: Backend> Ctx<B> {
    fn contains(&self, id: TensorId) -> bool {
        self.iter().any(|t| t.id == id)
    }
}

impl<B: Backend> core::ops::Deref for Ctx<B> {
    type Target = Vec<Tensor<B>>;

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
    fn type_name(&self) -> String {
        let full_name = std::any::type_name::<Self>().to_string();
        let splited: Vec<&str> = full_name.split(&['<', '>'][..]).collect();
        let function = splited[0]
            .split("::")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let backend = splited[1]
            .split("::")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        format!("{}<{}>", function.last().unwrap(), backend.last().unwrap())
    }
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B;
    fn backward(&mut self, grad: B) -> Grad<B>;
    fn parents_mut(&mut self) -> &mut Ctx<B>;
    fn parents_ref(&self) -> &Ctx<B>;
    fn apply(
        &mut self,
        x: &Tensor<B>,
        y: Option<&Tensor<B>>,
        z: Option<&Tensor<B>>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> Tensor<B>
    where
        Self: 'static + Sized,
    {
        // These steps are done before this. Function::default().apply()
        // self.device = device
        // self.needs_input_grad = [t.requires_grad for t in tensors]
        // self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        //
        let ctx = self;
        let ret_inner = ctx.forward(
            &x.inner,
            y.map(|t| &t.inner),
            z.map(|t| &t.inner),
            shape,
            const_,
        );
        let require_grad = x.require_grad
            || y.is_some_and(|t| t.require_grad)
            || z.is_some_and(|t| t.require_grad);
        // if self.require_grad: self.parents = tensors
        if require_grad {
            ctx.parents_mut().push(x.clone());
            if let Some(t) = y {
                ctx.parents_mut().push(t.clone());
            }
            if let Some(t) = z {
                ctx.parents_mut().push(t.clone());
            }
        }
        Tensor {
            inner: ret_inner,
            require_grad,
            _ctx: if require_grad {
                Some(dyn_clone::clone_box(&*ctx))
            } else {
                None
            },
            id: tensor_id(),
            grad: std::sync::Arc::default(),
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
    ($t: expr) => {{
        let n = B::Dtype::from_f32($t).unwrap();
        if n.is_nan() {
            panic!("your number is a NaN")
        }
        n
    }};
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        x.contiguous()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(grad)
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.x = Some(x.clone());
        x.sin()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = self.x.as_ref().unwrap();
        Grad::One(
            x.const_like(B::Dtype::from_f64(core::f64::consts::PI / 2.0).unwrap())
                .sub(x)
                .sin()
                .mul(&grad),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.x = Some(x.clone());
        x.log2()
            .mul(&x.const_like(df32!(2.0f32.log(core::f32::consts::E))))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(grad.div(self.x.as_ref().unwrap()))
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        let ret = x
            .mul(&x.const_like(df32!(1f32 / 2.0f32.log(core::f32::consts::E))))
            .exp2();
        self.ret = Some(ret.clone());
        ret
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(self.ret.as_ref().unwrap().mul(&grad))
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.ret = Some(x.sqrt());
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let ret = self.ret.as_ref().unwrap();
        Grad::One(grad.div(&ret.mul(&ret.const_like(df32!(2.0f32)))))
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}
//FIXME: Both Sum/Max reduce op is using a hack on shape param in forward.
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("Sum bwd should have a input_shape");
        if input_shape.len() > grad.shape().len() {
            let mut new_grad_shape = grad.shape();
            for _ in 0..input_shape.len() - grad.shape().len() {
                new_grad_shape.dims.push(1);
            }
            grad = grad.reshape(new_grad_shape);
        }
        Grad::One(
            grad.expand(
                self.input_shape
                    .as_ref()
                    .expect("Sum bwd should have a input_shape")
                    .clone(),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        let mut shape = shape.unwrap();
        if shape.len() == 1 {
            return x.sum(None, false);
        }
        let (keepdim, axis) = if shape.len() - x.shape().len() == 1 {
            //TODO: hack, need change
            (true, *shape.dims.last().unwrap())
        } else {
            (false, shape.dims.iter().position(|e| *e == 0).unwrap())
        };
        self.x = Some(x.clone());
        if !keepdim {
            panic!("please use reshape to remove dim")
        }
        self.ret = Some(x.max(Some(axis as isize), true));
        self.ret.as_ref().unwrap().clone()
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        let x_ref = self.x.as_ref().unwrap();
        let ret_ref = self.ret.as_ref().unwrap();
        let max_is_1s = x_ref
            .const_like(df32!(1.0))
            .sub(&x_ref.cmplt(&ret_ref.expand(x_ref.shape())));
        let mut div = max_is_1s.clone();
        for (i, (msh, gsh)) in max_is_1s
            .shape()
            .dims
            .iter()
            .zip(grad.shape().dims.iter())
            .enumerate()
        {
            if msh != gsh {
                div = div.sum(Some(i as isize), true);
            }
        }
        let div = div.expand(x_ref.shape());
        let grad_ret = max_is_1s.div(&div).mul(&grad.expand(x_ref.shape()));
        Grad::One(grad_ret)
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        x.cmplt(&y.expect("Less fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        unreachable!("Less op can not do backward pass")
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        x.sub(&y.expect("Sub fwd op expects rhs"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        let x = if self.need_input_grad[0] {
            Some(grad.clone())
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.ret = Some(
            x.const_like(df32!(1.0)).div(
                &x.const_like(df32!(1.0)).add(
                    &x.mul(&x.const_like(df32!(-1.0 / 2.0f32.log(core::f32::consts::E))))
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}
// --------------------------------- Tenary
#[derive(Clone, Debug)]
pub struct Where<B: Backend> {
    pub(crate) x: Option<B>,
    pub(crate) ctx: Ctx<B>,
    pub(crate) need_input_grad: [bool; 3],
}

impl<B: Backend> Default for Where<B> {
    fn default() -> Self {
        Self {
            need_input_grad: [false, false, false],
            x: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Where<B> {
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.x = Some(x.clone());
        x._where(
            y.as_ref().expect("Where fwd expects Y"),
            z.as_ref().expect("Where fwd expects Z"),
        )
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        // return None, \
        //        self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
        //        self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
        let x = self.x.as_ref().expect("where bwd should have x now");
        Grad::Two(
            if self.need_input_grad[1] {
                Some(x._where(&grad, &grad.const_like(B::Dtype::zero())))
            } else {
                None
            },
            if self.need_input_grad[2] {
                Some(x._where(&grad.const_like(B::Dtype::zero()), &grad))
            } else {
                None
            },
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}
// -------------------------------- Movement -------------------------------------
#[derive(Clone, Debug)]
pub struct Expand<B: Backend> {
    pub(crate) input_shape: Option<Shape>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Expand<B> {
    fn default() -> Self {
        Self {
            input_shape: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Expand<B> {
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        self.input_shape = Some(x.shape());
        x.expand(shape.expect("Expand mlops expect a shape"))
    }

    fn backward(&mut self, mut grad: B) -> Grad<B> {
        let grad_shape = grad.shape();
        for i in shape_to_axis(
            self.input_shape
                .as_ref()
                .expect("Expand bwd should have a shape now")
                .clone(),
            grad_shape,
        )
        .iter()
        {
            grad = grad.sum(Some(*i as isize), true);
        }
        Grad::One(grad)
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
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
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        mut shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        let order = shape.take().expect("Permute mlops expect a permute order");
        self.permute_order = Some(order.clone());
        x.permute(order)
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

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}

//NOTE: Pad/Shrink stores in Vec<(usize, usize)>, so we flatten that into a vec<usize> when using
//      this, such that we dont need a new param in this forwrad()
#[derive(Clone, Debug)]
pub struct Pad<B: Backend> {
    pub(crate) narg: Option<Vec<(usize, usize)>>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Pad<B> {
    fn default() -> Self {
        Self {
            narg: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Pad<B> {
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        let flatten_p = shape.unwrap();
        let mut narg = Vec::new();
        let mut arg = Vec::new();
        // windows sliding overlaps, so we skip 1, dont use skip, use step_by
        for (sh, p) in x
            .shape()
            .dims
            .iter()
            .zip(flatten_p.dims.windows(2).step_by(2))
        {
            narg.push((p[0], sh + p[0]));
            arg.push((p[0], p[1]));
        }
        assert!(
            narg.len() == x.shape().len(),
            "Pad fwd: Something is wrong when creating Vec<(usize, usize)>, padding:{:?} x:{}",
            narg,
            x.shape()
        );
        self.narg = Some(narg.clone());
        x.pad(arg, const_.expect("Pad fwd expect a const"))
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(
            grad.shrink(
                self.narg
                    .as_ref()
                    .expect("Reshape backward should already have a shape")
                    .clone(),
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[derive(Clone, Debug)]
pub struct Shrink<B: Backend> {
    pub(crate) narg: Option<Vec<(usize, usize)>>,
    pub(crate) ctx: Ctx<B>,
}

impl<B: Backend> Default for Shrink<B> {
    fn default() -> Self {
        Self {
            narg: None,
            ctx: Ctx::default(),
        }
    }
}

impl<B: Backend> Function<B> for Shrink<B> {
    fn forward(
        &mut self,
        x: &B,
        y: Option<&B>,
        z: Option<&B>,
        shape: Option<Shape>,
        const_: Option<B::Dtype>,
    ) -> B {
        let flatten_p = shape.unwrap();
        let mut narg = Vec::new();
        let mut padding = Vec::new();
        // windows sliding overlaps, so we skip 1, dont use skip, use step_by
        for (sh, p) in x
            .shape()
            .dims
            .iter()
            .zip(flatten_p.dims.windows(2).step_by(2))
        {
            narg.push((p[0], sh - p[1]));
            padding.push((p[0], p[1]));
        }
        assert!(
            narg.len() == x.shape().len(),
            "Pad fwd: Something is wrong when creating Vec<(usize, usize)>, padding:{:?} x:{}",
            narg,
            x.shape()
        );
        self.narg = Some(narg.clone());
        x.shrink(padding)
    }

    fn backward(&mut self, grad: B) -> Grad<B> {
        Grad::One(
            grad.pad(
                self.narg
                    .as_ref()
                    .expect("Reshape backward should already have a shape")
                    .clone(),
                B::Dtype::zero(), // WARN: This might be incorrect
            ),
        )
    }

    fn parents_mut(&mut self) -> &mut Ctx<B> {
        &mut self.ctx
    }

    fn parents_ref(&self) -> &Ctx<B> {
        &self.ctx
    }
}

#[test]
fn mlop_sin() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.sin();
    approx_eq!(
        x,
        [
            -0.841471,
            0.9092974,
            -0.14112,
            -0.7568025,
            0.9589243,
            -0.2794155,
            -0.6569866,
            0.98935825,
            -0.41211846
        ]
    );
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [
            0.54030222,
            -0.41614679,
            -0.9899925,
            -0.65364379,
            0.28366235,
            0.96017021,
            0.75390244,
            -0.14549987,
            -0.91113013
        ]
    );
}

#[test]
fn mlop_relu() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.relu();

    approx_eq!(x, [0., 2., 0., 4., 0., 6., 0., 8., 0.]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [0., 1., 0., 1., 0., 1., 0., 1., 0.]
    );

    let mut t = Tensor::<Cpu>::from_shape([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.relu();
    approx_eq!(x, [1., 0., 3., 0., 5., 0., 7., 0., 9.]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 0., 1., 0., 1., 0., 1., 0., 1.]
    );
}

#[test]
fn mlop_log() {
    let mut t = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.log();
    approx_eq!(
        x,
        [
            0., 0.6931472, 1.0986123, 1.3862944, 1.6094378, 1.7917595, 1.9459102, 2.0794415,
            2.1972246
        ]
    );
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 0.5, 0.33333333, 0.25, 0.2, 0.16666667, 0.14285714, 0.125, 0.11111111]
    );
}

#[test]
fn mlop_exp() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.exp();

    approx_eq!(
        x,
        [
            0.36787945,
            7.3890557,
            0.04978707,
            54.5981315,
            0.006737951,
            403.42868,
            0.00091188244,
            2980.9558,
            0.0001234099
        ]
    );
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [
            0.36787945,
            7.38905573,
            0.04978707,
            54.59813835,
            0.006737951,
            403.42868042,
            0.00091188244,
            2980.95586367,
            0.00012340993
        ]
    );
}

#[test]
fn mlop_sqrt() {
    let mut t = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.sqrt();
    approx_eq!(
        x,
        [1., 1.4142135, 1.7320508, 2., 2.236068, 2.4494898, 2.6457512, 2.828427, 3.]
    );
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [
            0.5, 0.3535534, 0.28867514, 0.25, 0.22360679, 0.20412414, 0.18898224, 0.1767767,
            0.16666667
        ]
    );
}

#[test]
fn mlop_sigmoid() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.sigmoid();
    approx_eq!(
        x,
        [
            0.26894143,
            0.880797,
            0.04742588,
            0.98201376,
            0.0066928547,
            0.9975274,
            0.0009110517,
            0.99966466,
            0.0001233947
        ]
    );
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [
            0.19661194,
            0.10499363,
            0.04517666,
            0.01766273,
            0.00664806,
            0.0024664658,
            0.00091022166,
            0.00033522327,
            0.00012337948
        ]
    );
}

#[test]
fn mlop_sum() {
    let mut t = Tensor::<Cpu>::from_shape(
        [
            0.26894143,
            0.880797,
            0.04742588,
            0.98201376,
            0.0066928547,
            0.9975274,
            0.0009110517,
            0.99966466,
            0.0001233947,
        ],
        [3, 3],
    );
    t.require_grad = true;
    let mut x = t.sum_all();
    approx_eq!(x, [4.184098]);
    x.backward();
    assert!(t.shape() == t.grad.lock().unwrap().as_ref().unwrap().shape());
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    );
}

#[test]
fn mlop_max() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.max_all();
    approx_eq!(x, [8.0]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [0., 0., 0., 0., 0., 0., 0., 1., 0.]
    );

    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.max(0);
    approx_eq!(x, [4.0, 8.0, 6.0]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [0., 0., 0., 1., 0., 1., 0., 1., 0.]
    );

    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.max(1);
    approx_eq!(x, [2.0, 6.0, 8.0]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [0., 1., 0., 0., 0., 1., 0., 1., 0.]
    );

    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    t.require_grad = true;
    let mut x = t.max(2);
    approx_eq!(x, [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    );
}

#[test]
fn mlop_less() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    let mut b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
    let mut x = t._lt(&b);
    approx_eq!(x, [1., 0., 1., 0., 1., 0., 1., 0., 1.]);
    // less has no bwd pass
}

#[test]
fn mlop_add() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    let mut b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
    t.require_grad = true;
    b.require_grad = true;
    let mut x = (&t + &b);
    approx_eq!(x, [0., 4., 0., 8., 0., 12., 0., 16., 0.]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    );
}

#[test]
fn mlop_sub() {
    let mut t = Tensor::<Cpu>::from_shape([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0], [3, 3]);
    let mut b = Tensor::<Cpu>::from_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
    t.require_grad = true;
    b.require_grad = true;
    let mut x = (&t - &b);
    approx_eq!(x, [-2., 0., -6., 0., -10., 0., -14., 0., -18.]);
    x.sum_all().backward();
    approx_eq!(
        t.grad.lock().unwrap().as_ref().unwrap(),
        [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    );
}

#[test]
fn mlop_mul() {
    let a = Tensor::<Cpu>::empty([3]).const_like(4.0);
    let b = Tensor::<Cpu>::empty([3]).const_like(0.5);
    let out = a * b;
    approx_eq!(out, [2.0, 2.0, 2.0]);
}
//
// #[test]
// fn mlop_div() {
//     todo!()
// }
//
// #[test]
// fn mlop_where() {
//     todo!()
// }
//
// #[test]
// fn mlop_expand() {
//     todo!()
// }
//
// #[test]
// fn mlop_reshape() {
//     todo!()
// }
//
// #[test]
// fn mlop_permute() {
//     todo!()
// }
//
// #[test]
// fn mlop_pad() {
//     todo!()
// }
//
// #[test]
// fn mlop_shrink() {
//     todo!()
// }
//
// #[test]
// fn mlop_flip() {
//     todo!()
// }
