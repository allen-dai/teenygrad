use opencl3::command_queue::{
    create_command_queue_with_properties, CommandQueue, CL_QUEUE_PROFILING_ENABLE,
};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_HALF_FLOAT};
use opencl3::program::Program;
use opencl3::types::{cl_double, cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

use crate::dtype::{DType, float32};
use crate::ops::Op;
use crate::renderer::cstyle::CstyleLanguage;

use super::{RawBuffer, RawBuf};

fn create_context() -> Context {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("No device found")
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device).expect("Context::from_device failed");
    context
}

// #[derive(Debug)]
// pub struct CLProgram {
//     program: Program,
//     renderer: CLRenderer,
//     buffe
// }
//
// impl CLProgram {
//     fn new(prg: &str) -> Result<Self> {
//         let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
//             .first()
//             .expect("no device found in platform");
//         let device = Device::new(device_id);
//
//         // Create a Context on an OpenCL device
//         let context = Context::from_device(&device).expect("Context::from_device failed");
//
//         // Create a command_queue on the Context's device
//         let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
//             .expect("CommandQueue::create_default failed");
//
//         // Build the OpenCL program source and create the kernel.
//         let program = Program::create_and_build_from_source(&context, prg, "")
//             .expect("Program::create_and_build_from_source failed");
//         Ok(Self { inner: program })
//     }
//
//     fn build(&mut self) {}
// }

#[derive(Debug, Clone)]
pub struct CLRenderer {
    pub renderer: CstyleLanguage,
}

#[derive(Debug)]
pub struct CLBuffer<T> {
    pub buf: Buffer<T>,
}

impl<T> CLBuffer<T> {
    fn new(size: usize) -> Self {
        let context = create_context();
        let buf =
            unsafe { Buffer::<T>::create(&context, CL_MEM_READ_WRITE, size, ptr::null_mut()).expect("unable to alloc buffer") };
        Self { buf }
    }
}

impl<T: std::fmt::Debug> RawBuf for CLBuffer<T> {}

impl Default for CLRenderer {
    fn default() -> Self {
        Self {
            renderer: CstyleLanguage {
                kernel_prefix: "__kernel ".into(),
                buffer_prefix: "__global ".into(),
                smem_align: "__attribute__ ((aligned (16))) ".into(),
                smem_prefix: "__local ".into(),
                arg_int_prefix: "const int".into(),
                half_prekernel: Some("#pragma OPENCL EXTENSION cl_khr_fp16 : enable".into()),
                barrier: "barrier(CLK_LOCAL_MEM_FENCE);".into(),
                float4: Some("(float4)".into()),
                gid: (0..3).map(|i| format!("get_group_id({i})")).collect(),
                lid: (0..3).map(|i| format!("get_local_id({i})")).collect(),
                uses_vload: true,
                ..Default::default()
            },
        }
    }
}

impl std::ops::Deref for CLRenderer {
    type Target = CstyleLanguage;

    fn deref(&self) -> &Self::Target {
        &self.renderer
    }
}

#[derive(Debug, Clone)]
pub struct CLDevice {
    renderer: CLRenderer,
}

#[test]
fn rawbuf() {
    let rb = RawBuffer::new(10, float32, CLBuffer::<f32>::new(32), "gpu");
}
