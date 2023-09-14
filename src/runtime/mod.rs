use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{backend::cpu::Cpu, dtype, prelude::Dtype};

pub mod opencl;

pub trait Runtime {}

pub trait RawBuf: Debug {}

#[derive(Clone, Debug)]
pub struct RawBuffer {
    size: usize,
    dtype: dtype::DType,
    buf: Arc<Mutex<dyn RawBuf>>,
    device: String,
}

impl RawBufferNew for RawBuffer {
    fn new<B: 'static + RawBuf>(size: usize, dtype: dtype::DType, buf: B, device: String) -> Self {
        Self {
            size,
            dtype,
            buf: Arc::new(Mutex::new(buf)),
            device,
        }
    }
}

pub trait RawBufferNew {
    fn new<B: 'static + RawBuf>(size: usize, dtype: dtype::DType, buf: B, device: String) -> Self;
}

pub trait RawBufferCopyFrom: RawBufferNew {
    fn _copy_from(&mut self, x: Cpu) {
        unimplemented!()
    }

    fn from_cpu() -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}
