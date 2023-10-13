use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{backend::cpu::Cpu, dtype, prelude::Dtype};

pub mod opencl;

pub trait Runtime {}

pub trait RawBuf: Debug {
    fn from_cpu() -> Self
    where
        Self: Sized,
    {
        unimplemented!()
    }
    fn to_cpu(&self) -> Cpu {
        unimplemented!()
    }
}

#[derive(Clone, Debug)]
pub struct RawBuffer {
    size: usize,
    dtype: dtype::DType,
    buf: Arc<Mutex<dyn RawBuf>>,
    device: String,
}

unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    fn new<B: 'static + RawBuf>(size: usize, dtype: dtype::DType, buf: B, device: &str) -> Self {
        Self {
            size,
            dtype,
            buf: Arc::new(Mutex::new(buf)),
            device: device.to_string(),
        }
    }
}
