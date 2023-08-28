#![allow(unused, unused_variables)]

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
pub mod tensor;
pub mod optim;

pub mod prelude {
    pub use rand::prelude::*;
    pub use crate::tensor::{shape::Shape, dtype::Dtype, core_ops::*};
    pub use crate::backend::cpu::Cpu;
    pub use crate::tensor::Tensor;
    pub use rand_distr::Standard;
    pub use num_traits::{FromPrimitive, ToPrimitive, Zero, One};
    pub use crate::backend::Backend;
    pub use crate::optim::{Optimizer, Adam};
}

use prelude::*;
