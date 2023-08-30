use num_traits::Float;

use crate::prelude::*;

pub trait Optimizer {
    fn zero_grad(&mut self);
    fn realize(&mut self);
    fn step(&mut self);
}

pub fn Adam<'a, B: Backend>(params: Vec<*mut Tensor<B>>, lr: f32) -> LAMP<B> {
    LAMP::new(params, lr, 0.9, 0.999, 1e-8, 0.0, true)
}

//def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
#[derive(Debug)]
pub struct LAMP<B: Backend> {
    pub(crate) params: Vec<*mut Tensor<B>>,
    pub(crate) buffers: Vec<*mut Tensor<B>>,
    pub(crate) lr: Tensor<B>, // on new() will take in a float, but it will make into a tensor
    pub(crate) b1: f32,
    pub(crate) b2: f32,
    pub(crate) eps: f32,
    pub(crate) wd: f32,
    pub(crate) adam: bool,
    pub(crate) t: f32,
    pub(crate) m: Vec<Tensor<B>>,
    pub(crate) v: Vec<Tensor<B>>,
}

impl<B: Backend> LAMP<B> {
    pub fn new(
        mut _params: Vec<*mut Tensor<B>>,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        wd: f32,
        adam: bool,
    ) -> Self {
        unsafe {
            let mut params = Vec::new();
            let mut buffers = Vec::new();
            while !_params.is_empty() {
                let t = _params.pop().unwrap();
                // if (*t).require_grad {
                //     params.push(t);
                // } else {
                //     buffers.push(t);
                // }
                (*t).require_grad = true;
                params.push(t);
            }
            let lr = Tensor::from_vec([B::Dtype::from_f32(lr).unwrap()], [1]);
            params.dedup_by_key(|t| (*(*t)).id);
            //buffers.dedup_by_key(|t| (*(*t)).id);
            let m = params
                .iter()
                .map(|t| Tensor::zeros((**t).shape()))
                .collect();
            let v = params
                .iter()
                .map(|t| Tensor::zeros((**t).shape()))
                .collect();
            let ret = Self {
                params,
                buffers,
                lr,
                b1,
                b2,
                eps,
                wd,
                adam,
                t: 0.,
                m,
                v,
            };
            //println!("{:?}", ret);
            ret
        }
    }
}

impl<B: Backend> Optimizer for LAMP<B> {
    fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            unsafe {
                let p = &(*(*p));
                *p.grad.lock().unwrap() = None;
            }
        }
    }

    fn realize(&mut self) {
        ()
    }

    fn step(&mut self) {
        self.t += 1.;
        unsafe {
            for (i, t) in self.params.iter_mut().enumerate() {
                let mut t = &mut (**t);
                assert!(t.grad.lock().unwrap().is_some());
                let g = (*t.grad.lock().unwrap()).clone();
                let mut g = g.unwrap();
                // self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g).realize()
                let mi = self.m[i].clone();
                self.m[i].assign(mi * self.b1 + &g * (1.0 - self.b1));
                // self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).realize()
                let b2v = &self.v[i] * self.b2;
                let g2 = (&g * &g) * (1.0 - self.b2);
                self.v[i].assign(b2v + g2);
                // m_hat = self.m[i] / (1.0 - self.b1**self.t)
                let m_hat = &self.m[i] / (1.0 - self.b1.powf(self.t));
                // println!("m_hat {:?}", m_hat.inner);
                // v_hat = self.v[i] / (1.0 - self.b2**self.t)
                let v_hat = &self.v[i] / (1.0 - self.b2.powf(self.t));
                // println!("v_hat {:?}", v_hat.inner);
                // up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
                let up = (m_hat / (v_hat.sqrt() + self.eps)) + &*t * self.wd;
                let r = if !self.adam { todo!() } else { 1.0 };
                t.assign(&*t - &(&self.lr * r * up));

                // just in case _ctx is attach in Function{}.apply()
                self.m[i]._ctx = None;
                self.v[i]._ctx = None;
                g._ctx = None;
            }
        }
        self.realize()
    }
}
