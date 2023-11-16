use crate::prelude::*;
use safetensors::{serialize_to_file, tensor::SafeTensors, SafeTensorError, View};

pub trait SafeTensor {
    fn from_file<P: AsRef<std::path::Path>>(
        &mut self,
        tensor_name: &str,
        safetensor_path: P,
    ) -> Result<(), SafeTensorError>;
    fn from_bytes(
        &mut self,
        tensor_name: &str,
        data: &[u8],
    ) -> Result<(), SafeTensorError>;
    fn to_safetensor<P: AsRef<std::path::Path>>(
        tensors: &[(&str, &Self)],
        path: P,
    ) -> Result<(), SafeTensorError>;
}

impl<B: Backend> SafeTensor for Tensor<B> {
    fn from_file<P: AsRef<std::path::Path>>(
        &mut self,
        tensor_name: &str,
        safetensor_path: P,
    ) -> Result<(), SafeTensorError> {
        let buffer = {
            let f = std::fs::File::open(safetensor_path)?;
            unsafe { memmap2::MmapOptions::new().map(&f)? }
        };
        let tensors = SafeTensors::deserialize(&buffer)?;
        let view = tensors.tensor(tensor_name)?;
        let data = view.data();
        assert!(
            self.shape() == view.shape().into(),
            "SafeTensor shape does not match tensor shape {} != {:?}",
            self.shape(),
            view.shape()
        );
        let nbytes = std::mem::size_of::<B::Dtype>();
        if data.as_ptr() as usize % nbytes == 0 {
            let content: &[B::Dtype] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const B::Dtype, data.len() / nbytes)
            };
            self.assign(Self::from_shape(content, self.shape()));
        } else {
            let mut content: Vec<B::Dtype> = Vec::with_capacity(data.len() / nbytes);
            let mut i = 0;
            while i < data.len() {
                content.push(B::Dtype::from_le_bytes(&data[i..i + nbytes]));
                i += nbytes;
            }
            self.assign(Self::from_shape(content, self.shape()));
        }
        Ok(())
    }

    fn from_bytes(
        &mut self,
        tensor_name: &str,
        data: &[u8],
    ) -> Result<(), SafeTensorError> {
        let tensors = SafeTensors::deserialize(&data)?;
        let view = tensors.tensor(tensor_name)?;
        let data = view.data();
        assert!(
            self.shape() == view.shape().into(),
            "SafeTensor shape does not match tensor shape {} != {:?}",
            self.shape(),
            view.shape()
        );
        let nbytes = std::mem::size_of::<B::Dtype>();
        if data.as_ptr() as usize % nbytes == 0 {
            let content: &[B::Dtype] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const B::Dtype, data.len() / nbytes)
            };
            self.assign(Self::from_shape(content, self.shape()));
        } else {
            let mut content: Vec<B::Dtype> = Vec::with_capacity(data.len() / nbytes);
            let mut i = 0;
            while i < data.len() {
                content.push(B::Dtype::from_le_bytes(&data[i..i + nbytes]));
                i += nbytes;
            }
            self.assign(Self::from_shape(content, self.shape()));
        }
        Ok(())
    }

    fn to_safetensor<P: AsRef<std::path::Path>>(
        tensors: &[(&str, &Self)],
        path: P,
    ) -> Result<(), SafeTensorError> {
        serialize_to_file(
            tensors
                .iter()
                .map(|(name, t)| (*name, TensorView::from(*t)))
                .collect::<Vec<(&str, TensorView)>>(),
            &None,
            path.as_ref(),
        )?;
        Ok(())
    }
}
// a wrapper for tensor. view trait needs a borrowed &[u8] for shape
// beacause tensor doesnt store any info except the backend buffer
//
// We can get rid of this when we have shapetracker
struct TensorView {
    dtype: String,
    shape: Shape,
    data_bytes: Vec<u8>,
}

impl<B: Backend> From<&Tensor<B>> for TensorView {
    fn from(value: &Tensor<B>) -> Self {
        Self {
            dtype: value.dtype(),
            shape: value.shape(),
            data_bytes: value
                .to_vec()
                .iter()
                .flat_map(|f| f._to_le_bytes())
                .collect(),
        }
    }
}

impl View for TensorView {
    fn dtype(&self) -> safetensors::Dtype {
        match self.dtype.as_str() {
            "f32" => safetensors::Dtype::F32,
            "f64" => safetensors::Dtype::F64,
            d => panic!("unsupported dtype: {d}"),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape.dims
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        std::borrow::Cow::from(&self.data_bytes)
    }

    fn data_len(&self) -> usize {
        self.data_bytes.len()
    }
}

#[test]
fn safetensor_loadsave() {
    let original = Tensor::<Cpu>::scaled_uniform([32, 16, 4, 16]);
    let y = Tensor::<Cpu>::scaled_uniform([32, 16, 4, 16]);
        Tensor::to_safetensor(&[("a", &original)], "/tmp/sttest.safetensors")
        .unwrap();
    let mut loaded = Tensor::<Cpu>::empty([32, 16, 4, 16]);
    loaded
        .from_file("a", "/tmp/sttest.safetensors")
        .unwrap();
    assert!((loaded * &y).to_vec() == (original * &y).to_vec());
}
