use kdam::{tqdm, BarExt};
use rand::{seq::SliceRandom, thread_rng};
use teenygrad::prelude::*;

pub fn main() {
    let training = true;
    let mut generator = LinearGen::<Cpu>::default();
    let mut discriminator = LinearDisc::<Cpu>::default();
    if generator
        .load("./weights/mnistgan_gen.safetensors")
        .is_err()
    {
        println!("couldn't find generator safetensor, ignoring...")
    }
    if training {
        let mut rng = thread_rng();
        if discriminator
            .load("./weights/mnistgan_disc.safetensors")
            .is_err()
        {
            println!("couldn't find discriminator safetensor, ignoring...")
        };
        let out_noise = Tensor::<Cpu>::randn([16, 128]);
        let (epochs, batch_size, k) = (100, 300, 1);
        let n_steps = 100;
        let (train_img, _, _, _) = fetch_mnist_shuffled::<Cpu>(batch_size);
        let num_batch = train_img.len();
        let mut gen_optim = teenygrad::nn::optim::adam_with(
            &[
                &mut generator.l1,
                &mut generator.l2,
                &mut generator.l3,
                &mut generator.l4,
            ],
            &[0.0002, 0.5],
        );

        let mut disc_optim = teenygrad::nn::optim::adam_with(
            &[
                &mut discriminator.l1,
                &mut discriminator.l2,
                &mut discriminator.l3,
                &mut discriminator.l4,
            ],
            &[0.0002, 0.5],
        );

        let mut pb = tqdm!(total = epochs);
        pb.set_description(format!(
            "Generator loss: {:.2} | Discriminator loss: {:.2}",
            0, 0
        ));
        pb.refresh().unwrap();
        for epoch in 0..epochs {
            let (mut loss_g, mut loss_d) = (0f32, 0f32);
            for _ in 0..n_steps {
                let real_data = Tensor::<Cpu>::from(train_img[rng.gen_range(0..num_batch)].clone())
                    .reshape([batch_size, 28 * 28])
                    / (255.0 / 2.0)
                    - 1.0;
                for _ in 0..k {
                    // Train Discriminator
                    let noise = Tensor::<Cpu>::randn([batch_size, 128]);
                    let fake_data = generator.forward(&noise).detach();
                    let real_labels = make_labels::<Cpu>(batch_size, 1);
                    let fake_labels = make_labels::<Cpu>(batch_size, 0);
                    disc_optim.zero_grad();
                    let output_real = discriminator.forward(&real_data);
                    let output_fake = discriminator.forward(&fake_data);
                    let mut loss_real = (output_real * real_labels).mean();
                    let mut loss_fake = (output_fake * fake_labels).mean();
                    loss_real.backward();
                    loss_fake.backward();
                    disc_optim.step();
                    loss_d = loss_d + loss_real.to_vec()[0] + loss_fake.to_vec()[0];
                }
                // Train Generator
                gen_optim.zero_grad();
                let real_labels = make_labels::<Cpu>(batch_size, 1);
                let noise = Tensor::<Cpu>::randn([batch_size, 128]);
                let fake_data = generator.forward(&noise);
                let out = discriminator.forward(&fake_data);
                let mut loss = (out * real_labels).mean();
                loss.backward();
                gen_optim.step();
                loss_g = loss_g + loss.to_vec()[0];
            }

            let mut fake_images = (generator.forward(&out_noise).detach() + 1f32) / 2f32;
            fake_images = fake_images * 255.0;
            // Just reshaping into a 4*28 by 4*28 in a contigous array. so image can save it in
            // that shape without distortion.
            fake_images = fake_images.reshape([4,4,28,28]).permute([0,1,3,2]).reshape([4, 4*28, 28]).transpose(1, 2);
            let data = fake_images
                .to_vec()
                .iter()
                .map(|e| *e as u8)
                .collect::<Vec<u8>>();
            image::save_buffer_with_format(
                &std::path::Path::new(&format!("./images/image-{epoch}.jpg")),
                &data,
                (4*28) as u32,
                (4*28) as u32,
                image::ColorType::L8,
                image::ImageFormat::Jpeg,
            )
            .unwrap();

            pb.set_description(format!(
                "Generator loss: {:.8} | Discriminator loss: {:.8}",
                loss_g / batch_size as f32,
                loss_d / batch_size as f32
            ));
            pb.update(1).unwrap();
            if (epoch + 1) % 20 == 0 {
                generator
                    .save("./weights/mnistgan_gen.safetensors")
                    .unwrap();
                discriminator
                    .save("./weights/mnistgan_disc.safetensors")
                    .unwrap();
            }
        }
    }
}

pub struct LinearGen<B: Backend> {
    pub l1: Tensor<B>,
    pub l2: Tensor<B>,
    pub l3: Tensor<B>,
    pub l4: Tensor<B>,
}

impl<B: Backend> Default for LinearGen<B> {
    fn default() -> Self {
        Self {
            l1: Tensor::scaled_uniform([128, 256]),
            l2: Tensor::scaled_uniform([256, 512]),
            l3: Tensor::scaled_uniform([512, 1024]),
            l4: Tensor::scaled_uniform([1024, 784]),
        }
    }
}

impl<B: Backend> LinearGen<B> {
    fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        let mut x = x.matmul(&self.l1).leakyrelu(Some(0.2));
        x = x.matmul(&self.l2).leakyrelu(Some(0.2));
        x = x.matmul(&self.l3).leakyrelu(Some(0.2));
        x = x.matmul(&self.l4).tanh();
        x
    }

    fn save(&self, path: &str) -> Result<(), safetensors::SafeTensorError> {
        Tensor::to_safetensor(
            &[
                ("l1", &self.l1),
                ("l2", &self.l2),
                ("l3", &self.l3),
                ("l4", &self.l4),
            ],
            path,
        )?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), safetensors::SafeTensorError> {
        self.l1.from_safetensor("l1", path)?;
        self.l2.from_safetensor("l2", path)?;
        self.l3.from_safetensor("l3", path)?;
        self.l4.from_safetensor("l4", path)?;
        Ok(())
    }
}

pub struct LinearDisc<B: Backend> {
    pub l1: Tensor<B>,
    pub l2: Tensor<B>,
    pub l3: Tensor<B>,
    pub l4: Tensor<B>,
}

impl<B: Backend> Default for LinearDisc<B> {
    fn default() -> Self {
        Self {
            l1: Tensor::scaled_uniform([784, 1024]),
            l2: Tensor::scaled_uniform([1024, 512]),
            l3: Tensor::scaled_uniform([512, 256]),
            l4: Tensor::scaled_uniform([256, 2]),
        }
    }
}

impl<B: Backend> LinearDisc<B> {
    fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        let mut x = (x.matmul(&self.l1) + 1.0f32)
            .leakyrelu(Some(0.2))
            .dropout(Some(0.3));
        x = x.matmul(&self.l2).leakyrelu(Some(0.2)).dropout(Some(0.3));
        x = x.matmul(&self.l3).leakyrelu(Some(0.2)).dropout(Some(0.3));
        x = x.matmul(&self.l4).log_softmax();
        x
    }

    fn save(&self, path: &str) -> Result<(), safetensors::SafeTensorError> {
        Tensor::to_safetensor(
            &[
                ("l1", &self.l1),
                ("l2", &self.l2),
                ("l3", &self.l3),
                ("l4", &self.l4),
            ],
            path,
        )?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), safetensors::SafeTensorError> {
        self.l1.from_safetensor("l1", path)?;
        self.l2.from_safetensor("l2", path)?;
        self.l3.from_safetensor("l3", path)?;
        self.l4.from_safetensor("l4", path)?;
        Ok(())
    }
}

fn fetch_mnist_shuffled<B: Backend>(
    batch_size: usize,
) -> (
    Vec<Vec<B::Dtype>>,
    Vec<Vec<B::Dtype>>,
    Vec<Vec<B::Dtype>>,
    Vec<Vec<B::Dtype>>,
) {
    use mnist::Mnist;
    let mnist = Mnist::from_download().expect("mnist download failed");
    let Mnist {
        train_images,
        train_labels,
        test_images,
        test_labels,
    } = mnist;
    let mut rng = thread_rng();

    // batching train
    let mut shuffle_idx: Vec<usize> = (0..60000).collect();
    shuffle_idx.shuffle(&mut rng);
    let mut train_img_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(60000 * 28 * 28);
    let mut train_lbl_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(60000 * 10);
    let mut tain_img_in_one_batch = Vec::with_capacity(batch_size);
    let mut train_lbl_in_one_batch = Vec::with_capacity(batch_size);
    for i in 0..60000 {
        for ii in 0..28 * 28 {
            tain_img_in_one_batch
                .push(B::Dtype::from_u8(train_images[(shuffle_idx[i] * (28 * 28)) + ii]).unwrap());
        }
        train_lbl_in_one_batch.push(B::Dtype::from_u8(train_labels[shuffle_idx[i]]).unwrap());
        if (i + 1) % batch_size == 0 {
            train_img_batched.push(tain_img_in_one_batch.drain(..).collect::<Vec<B::Dtype>>());
            train_lbl_batched.push(train_lbl_in_one_batch.drain(..).collect::<Vec<B::Dtype>>());
        }
    }
    // batching test
    let mut shuffle_idx: Vec<usize> = (0..10000).collect();
    shuffle_idx.shuffle(&mut rng);
    let mut test_img_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(10000 * 28 * 28);
    let mut test_lbl_batched: Vec<Vec<B::Dtype>> = Vec::with_capacity(10000 * 10);
    let mut test_img_in_one_batch = Vec::with_capacity(batch_size);
    let mut test_lbl_in_one_batch = Vec::with_capacity(batch_size);
    for i in 0..10000 {
        for ii in 0..28 * 28 {
            test_img_in_one_batch
                .push(B::Dtype::from_u8(test_images[(shuffle_idx[i] * (28 * 28)) + ii]).unwrap());
        }
        test_lbl_in_one_batch.push(B::Dtype::from_u8(test_labels[shuffle_idx[i]]).unwrap());
        if (i + 1) % batch_size == 0 {
            test_img_batched.push(test_img_in_one_batch.drain(..).collect::<Vec<B::Dtype>>());
            test_lbl_batched.push(test_lbl_in_one_batch.drain(..).collect::<Vec<B::Dtype>>());
        }
    }
    (
        train_img_batched,
        train_lbl_batched,
        test_img_batched,
        test_lbl_batched,
    )
}

fn make_labels<B: Backend>(bs: usize, col: usize) -> Tensor<B> {
    let val = -2.0f64;
    let ret = if col == 1 {
        Tensor::<B>::_arange(val - val, val + val, val).reshape([1, 2])
    } else {
        Tensor::<B>::_arange(val, val * -1.0, val * -1.0).reshape([1, 2])
    };
    ret.expand([bs, 2])
}
