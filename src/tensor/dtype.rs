use half::{bf16, f16};
use num_traits::{FromPrimitive, NumOps, One, ToPrimitive, Zero, Float};

pub trait Dtype:
    'static
    + core::fmt::Debug
    + Default
    + Copy
    + Send
    + Sync
    + FromPrimitive
    + ToPrimitive
    + NumOps
    + One
    + Zero
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
    + core::ops::Add<Self, Output = Self>
    + core::ops::Sub<Self, Output = Self>
    + core::ops::Mul<Self, Output = Self>
    + core::ops::Div<Self, Output = Self>
    + PartialEq
    + PartialOrd
    + Float
{
    const PI: Self;
}

impl Dtype for f64 {
    const PI: Self = std::f64::consts::PI;
}
impl Dtype for f32 {
    const PI: Self = std::f32::consts::PI;
}
// impl Dtype for f16 {}
// impl Dtype for bf16 {}
// impl Dtype for i64 {}
// impl Dtype for i32 {}
// impl Dtype for i16 {}
// impl Dtype for i8 {}
// impl Dtype for u64 {}
// impl Dtype for u32 {}
// impl Dtype for u16 {}
// impl Dtype for u8 {}

macro_rules! df {
    ($t: expr) => {
        Dtype::from_f64($t).unwrap()
    };
}


macro_rules! di {
    ($t: expr) => {
        Dtype::from_isize($t).unwrap()
    };
}


macro_rules! du {
    ($t: expr) => {
        Dtype::from_usize($t).unwrap()
    };
}

pub(crate) use df;
pub(crate) use di;
pub(crate) use du;
