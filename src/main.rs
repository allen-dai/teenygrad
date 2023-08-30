use std::time::SystemTime;
use teenygrad::prelude::*;

fn main() {
    let mut net = ConvNet::<Cpu>::default();
    //println!("{}", net.c1);
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
    let seven: Vec<f32> = vec![
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 83., 130.,
        130., 155., 194., 163., 130., 130., 231., 255., 255., 95., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 92., 253., 253., 253., 253., 253., 253., 253., 253., 253.,
        253., 253., 207., 13., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 149., 253.,
        253., 253., 253., 253., 253., 253., 253., 253., 253., 253., 188., 11., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 62., 105., 123., 228., 204., 105., 105., 105., 105.,
        225., 253., 253., 30., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 205., 253., 253., 30., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11., 212., 253., 240., 26., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 156., 253.,
        238., 72., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 47., 236., 253., 166., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 96., 253., 253., 139., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 16., 207., 253., 233., 32.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        156., 253., 253., 79., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 17., 206., 253., 231., 35., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 56., 253., 253., 179., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 18., 184.,
        253., 230., 57., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 50., 253., 253., 139., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 99., 253., 150., 8., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 152., 246., 253.,
        68., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 90., 247., 253., 201., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 30., 249., 253., 253., 79., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 32., 253., 171., 105., 2., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ];

    let mut x = Tensor::from_vec(
        seven
            .iter()
            .map(|e| B::Dtype::from_f32(*e).unwrap())
            .collect::<Vec<B::Dtype>>(),
        [1, 28 * 28],
    );
    let mut y = vec![B::Dtype::zero(); 10];
    y[6] = B::Dtype::one();
    let mut y = Tensor::from_vec(y, [10]);
    //println!("{}", x);

    let mut optim = Adam(vec![&mut model.c1, &mut model.c2, &mut model.l1], 0.001);

    let mut it = 1f64;
    let s = SystemTime::now();
    for _ in 0..50 {
        let out = model.forward(&x);
        println!("{:?}", out.inner);
        let mut loss = out.sparse_categorical_crossentropy(&y);
        optim.zero_grad();
        loss.backward();
        optim.step();
        let e = SystemTime::now();
        let t = e.duration_since(s).unwrap().as_secs_f64();
        it += 1.0;
        println!("loss: {:?}", loss.inner);
        println!("{}it/s", it / t);
    }
}
