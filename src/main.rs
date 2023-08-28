use std::time::SystemTime;
use teenygrad::prelude::*;

fn main() {
    let mut net = ConvNet::<Cpu>::default();
    train(&mut net);
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

fn train<B: Backend>(model: &mut ConvNet<B>) {
    let mut x = Tensor::ones([28 * 28]);
    let mut y = vec![B::Dtype::zero();10];
    y[5] = B::Dtype::one();
    let mut y = Tensor::from_vec(y, [10]);

    let mut optim = Adam(vec![&mut model.c1, &mut model.c2, &mut model.l1], 0.001);

    let mut it = 1f64;

    let s = SystemTime::now();
    for _ in 0..10 {
        let out = model.forward(&x);
        let mut loss = out.sparse_categorical_crossentropy(&y);
        optim.zero_grad();
        loss.backward();
        println!("{:?}", loss.inner);
        //println!("{:?}", loss._ctx);
        optim.step();

        //println!("{}", loss);
        let e = SystemTime::now();
        let t = e.duration_since(s).unwrap().as_secs_f64();
        println!("{}it/s", it / t);
        it += 1.0;
    }
}
