use rand_distr::StandardNormal;
use std::time::{Duration, SystemTime};
use teenygrad::prelude::*;

fn main() {
    let mut x = Tensor::<Cpu>::randn([128, 1, 28, 28]);
    let inter_chan = 8;
    let out_chan = 16;
    let inter_chan = 8;
    let out_chan = 16;
    let conv = 3;
    let mut c1 = Tensor::<Cpu>::randn([inter_chan, 1, conv, conv]);
    let mut c2 = Tensor::<Cpu>::randn([out_chan, inter_chan, conv, conv]);
    let mut l1 = Tensor::<Cpu>::randn([out_chan * 5 * 5, 10]);
    //println!("{a} {b}");
    let s = SystemTime::now();
    let mut it = 1f64;
    loop {
        let c = fwd(x.clone());
        let e = SystemTime::now();
        let t = e.duration_since(s).unwrap().as_secs_f64();
        println!("{}it/s", it / t);
        it += 1.0;
    }
}

fn fwd<B: Backend>(mut x: Tensor<B>)
{
    let inter_chan = 8;
    let out_chan = 16;
    let conv = 3;
    let mut c1 = Tensor::randn([inter_chan, 1, conv, conv]);
    let mut c2 = Tensor::randn([out_chan, inter_chan, conv, conv]);
    let mut l1 = Tensor::randn([out_chan * 5 * 5, 10]);
    x = x.conv2d(&c1, None, 1, 1, 1, 0);
    x = x.conv2d(&c2, None, 1, 1, 1, 0);
    x.matmul(&l1);
}
