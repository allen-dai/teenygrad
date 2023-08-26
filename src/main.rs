use std::time::{Duration, SystemTime};
use teenygrad::prelude::*;

fn main() {
    let m = 200;
    let n = 200;
    let k = 200;
    let mut a =
        Tensor::<Cpu>::randn([m, n]);
    let mut b =
        Tensor::<Cpu>::randn([n, k]);
    //println!("{a} {b}");
    let s = SystemTime::now();
    let c = a.matmul(&b);
    let e = SystemTime::now();
    let flops = (n as f64).powf(3.0) * 2.0;
    println!(
        "{:?} GFLOPS",
        flops / e.duration_since(s).unwrap().as_secs_f64() / 1e9
        );
}
