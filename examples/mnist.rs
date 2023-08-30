use kdam::{tqdm, BarExt};
use rand::{seq::SliceRandom, thread_rng};
use teenygrad::prelude::*;

pub fn main() {
    let mut model = ConvNet::<Cpu>::default();
    train(&mut model);
}
pub struct ConvNet<B: Backend> {
    pub c1: Tensor<B>,
    pub c2: Tensor<B>,
    pub l1: Tensor<B>,
}

impl<B: Backend> Default for ConvNet<B> {
    fn default() -> Self {
        let conv = 3;
        let cin = 8;
        let cout = 16;
        Self {
            c1: Tensor::scaled_uniform([cin, 1, conv, conv]),
            c2: Tensor::scaled_uniform([cout, cin, conv, conv]),
            l1: Tensor::scaled_uniform([cout * 5 * 5, 10]),
        }
    }
}

impl<B: Backend> ConvNet<B> {
    fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        let d1 = x.shape().numel() / 28 / 28;
        let mut y = x.reshape([d1, 1, 28, 28]);
        y = y.conv2d(&self.c1).relu().max_pool2d();
        y = y.conv2d(&self.c2).relu().max_pool2d();
        y = y.reshape([y.shape()[0], y.shape().numel() / y.shape()[0]]);
        y = y.matmul(&self.l1).log_softmax();
        y
    }
}

fn train<B: Backend>(model: &mut ConvNet<B>) -> Result<(), Box<dyn std::error::Error>> {
    use mnist::Mnist;
    let mnist = Mnist::from_download()?;
    let Mnist {
        train_images,
        train_labels,
        test_images,
        test_labels,
    } = mnist;
    let mut optim = adam(vec![&mut model.c1, &mut model.c2, &mut model.l1], 0.001);
    let mut rng = thread_rng();
    let mut shuffle_idx: Vec<usize> = (0..60000).collect();
    let batch_size = 128;
    shuffle_idx.shuffle(&mut rng);
    let mut img_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(60000 * 28 * 28);
    let mut lbl_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(60000 * 10);
    let mut img_in_one_batch = Vec::with_capacity(batch_size);
    let mut lbl_in_one_batch = Vec::with_capacity(batch_size);
    for i in 0..60000 {
        for ii in 0..28 * 28 {
            img_in_one_batch
                .push(B::Dtype::from_u8(train_images[(shuffle_idx[i] * (28 * 28)) + ii]).unwrap());
        }
        lbl_in_one_batch.push(B::Dtype::from_u8(train_labels[shuffle_idx[i]]).unwrap());
        if (i + 1) % batch_size == 0 {
            img_batched.push(img_in_one_batch.clone());
            lbl_batched.push(lbl_in_one_batch.clone());
            img_in_one_batch.clear();
            lbl_in_one_batch.clear();
        }
    }
    let mut pb = tqdm!(total=100);
    pb.set_description(format!("loss: {:.05}", 0));
    pb.refresh()?;
    for i in 0..100 {
        let x = Tensor::from_vec(&*img_batched[i], [batch_size, 1, 28, 28]);
        let y = Tensor::from_vec(&*lbl_batched[i], [batch_size]);
        let out = model.forward(&x);
        let mut loss = out.sparse_categorical_crossentropy(&y);
        optim.zero_grad();
        loss.backward();
        optim.step();
        pb.set_description(format!("loss: {:.05?}", loss.inner.to_vec()[0]));
        pb.update(1)?;
    }
    Ok(())
}
