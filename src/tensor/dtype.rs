pub trait Dtype:
    'static
    + core::fmt::Debug
    + Default
    + Copy
    + Send
    + Sync
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
    + num_traits::NumOps
    + num_traits::One
    + num_traits::Zero
    + num_traits::Float
    + FromBytes
    + ToBytes
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
{
}

impl Dtype for f64 {}
impl Dtype for f32 {}
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

pub trait FromBytes {
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn from_be_bytes(bytes: &[u8]) -> Self;
    fn from_ne_bytes(bytes: &[u8]) -> Self;
}

macro_rules! impl_from_bytes {
    ($t:ty) => {
        impl FromBytes for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }

            fn from_be_bytes(bytes: &[u8]) -> Self {
                Self::from_be_bytes(bytes.try_into().unwrap())
            }

            fn from_ne_bytes(bytes: &[u8]) -> Self {
                Self::from_ne_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

impl_from_bytes!(f64);
impl_from_bytes!(f32);

pub trait ToBytes {
    type BytesArray: IntoIterator<Item = u8>;
    fn _to_le_bytes(&self) -> Self::BytesArray;
    fn _to_be_bytes(&self) -> Self::BytesArray;
    fn _to_ne_bytes(&self) -> Self::BytesArray;
}

macro_rules! impl_to_bytes {
    ($t:ty, $a:ty) => {
        impl ToBytes for $t {
            type BytesArray = $a;
            fn _to_le_bytes(&self) -> Self::BytesArray {
                self.to_le_bytes()
            }
            fn _to_be_bytes(&self) -> Self::BytesArray {
                self.to_be_bytes()
            }
            fn _to_ne_bytes(&self) -> Self::BytesArray {
                self.to_ne_bytes()
            }
        }
    };
}

impl_to_bytes!(f32, [u8; 4]);
impl_to_bytes!(f64, [u8; 8]);
