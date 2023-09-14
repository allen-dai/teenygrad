#![allow(non_upper_case_globals)]

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DType {
    pub priority: usize,
    pub itemsize: usize,
    pub name: &'static str,
    pub sz: usize,
    pub shape: Option<Vec<isize>>,
    pub ptr: bool,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sh) = &self.shape {
            return write!(f, "dtypes.{}({:?})", self.name, sh);
        } else if self.ptr {
            return write!(f, "ptr.{}", self.name);
        }
        write!(f, "dtypes.{}", self.name)
    }
}

impl DType {
    pub fn is_int(&self) -> bool {
        matches!(
            *self,
            int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            *self,
            float16 | float32 | float64 | _half4 | _float2 | _float4
        )
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(*self, uint8 | uint16 | uint32 | uint64)
    }
}

pub const _bool: DType = DType {
    priority: 0,
    itemsize: 1,
    name: "bool",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const float16: DType = DType {
    priority: 0,
    itemsize: 2,
    name: "half",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const half: DType = float16;

pub const float32: DType = DType {
    priority: 4,
    itemsize: 4,
    name: "float",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const float_: DType = float32;

pub const float64: DType = DType {
    priority: 0,
    itemsize: 8,
    name: "double",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const double: DType = float64;

pub const int8: DType = DType {
    priority: 0,
    itemsize: 1,
    name: "char",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int16: DType = DType {
    priority: 1,
    itemsize: 2,
    name: "short",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int32: DType = DType {
    priority: 2,
    itemsize: 4,
    name: "int",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const int64: DType = DType {
    priority: 3,
    itemsize: 8,
    name: "long",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint8: DType = DType {
    priority: 0,
    itemsize: 1,
    name: "unsigned char",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint16: DType = DType {
    priority: 1,
    itemsize: 2,
    name: "unsigned short",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint32: DType = DType {
    priority: 2,
    itemsize: 4,
    name: "unsigned int",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const uint64: DType = DType {
    priority: 3,
    itemsize: 8,
    name: "unsigned long",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const bfloat16: DType = DType {
    priority: 0,
    itemsize: 2,
    name: "__bf16",
    sz: 1,
    shape: None,
    ptr: false,
};

pub const _int2: DType = DType {
    priority: 2,
    itemsize: 8,
    name: "int2",
    sz: 2,
    shape: None,
    ptr: false,
};

pub const _half4: DType = DType {
    priority: 0,
    itemsize: 8,
    name: "half4",
    sz: 4,
    shape: None,
    ptr: false,
};

pub const _float2: DType = DType {
    priority: 4,
    itemsize: 8,
    name: "float2",
    sz: 2,
    shape: None,
    ptr: false,
};

pub const _float4: DType = DType {
    priority: 4,
    itemsize: 16,
    name: "float4",
    sz: 4,
    shape: None,
    ptr: false,
};

pub const _arg_int32: DType = DType {
    priority: 2,
    itemsize: 4,
    name: "_arg_int32",
    sz: 1,
    shape: None,
    ptr: false,
};
