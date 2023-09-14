#![allow(dead_code, unused)]

use std::sync::{Arc, Mutex};

lazy_static::lazy_static! {
    pub static ref RNG: Arc<Mutex<StdRng>> = {
        let start = std::time::SystemTime::now();
        let time = start
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
            Arc::new(Mutex::new(StdRng::seed_from_u64(time as u64)))
    };
}

pub mod backend;
pub mod nn;
pub mod tensor;
pub mod util;
pub mod lazy;
pub mod shape;
pub mod renderer;
pub mod dtype;
pub mod codegen;
pub mod ops;
pub mod runtime;

pub mod prelude {
    pub use crate::backend::cpu::Cpu;
    pub use crate::backend::Backend;
    pub use crate::nn::optim::{adam, Optimizer};
    pub use crate::tensor::Tensor;
    pub use crate::tensor::{core_ops::*, dtype::{Dtype, FromBytes, ToBytes}, shape::Shape};
    pub use num_traits::{FromPrimitive,  ToPrimitive, One, Zero, Float};
    pub use rand::prelude::*;
    pub use rand_distr::Standard;
    pub use crate::util::safetensor::SafeTensor;
    pub(crate) use crate::tensor::mlops::*;
    pub(crate) use crate::approx_eq;
    pub use crate::view;
}

use prelude::*;
