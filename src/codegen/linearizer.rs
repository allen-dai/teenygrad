use crate::{dtype, ops::OpType, lazy::LazyBuffer};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UOps {
    LOOP,
    END,
    SPECIAL,
    DEFINE_GLOBAL,
    DEFINE_LOCAL,
    DEFINE_ACC,
    LOAD,
    STORE,
    CONST,
    BARRIER,
    ALU,
    WMMA,
    CAST,
    GEP,
}

#[derive(Clone, Debug, Hash, Eq)]
pub enum Args {
    Str(String),
    Op(OpType),
    Buf((LazyBuffer, dtype::DType)),
    Int(isize),
}

impl Args {
    pub fn to_str(&self) -> String {
        match self {
            Args::Str(s) => s.clone(),
            t => panic!("Can not to_str() {t:?}"),
        }
    }

    pub fn to_op(&self) -> OpType {
        match self {
            Args::Op(op) => op.clone(),
            t => panic!("Can not to_op() {t:?}"),
        }
    }

    pub fn to_buf(&self) -> (LazyBuffer, dtype::DType) {
        match self {
            Args::Buf(buf) => buf.clone(),
            t => panic!("Can not to_buf() {t:?}"),
        }
    }

    pub fn to_int(&self) -> isize {
        match self {
            Args::Int(i) => i.clone(),
            t => panic!("Can not to_buf() {t:?}"),
        }
    }
}

impl PartialEq for Args {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl PartialEq<str> for Args {
    fn eq(&self, other: &str) -> bool {
        match self {
            Args::Str(s) => s == other,
            _ => false,
        }
    }
}

impl PartialEq<OpType> for Args {
    fn eq(&self, other: &OpType) -> bool {
        match self {
            Args::Op(op) => op == other,
            _ => false,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct UOp {
    pub(crate) uop: UOps,
    pub(crate) dtype: Option<dtype::DType>,
    pub(crate) vin: Vec<UOp>,
    pub(crate) args: Vec<Args>,
    pub(crate) num: usize,
}

impl core::fmt::Display for UOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.4} {:<20?}: {} {:<32?} {:?}",
            self.num,
            self.uop,
            if self.dtype.is_some() {
                format!("{:?}", self.dtype.as_ref().unwrap())
            } else {
                format!("{:<25}", "")
            },
            self.vin.iter().map(|x| x.num).collect::<Vec<usize>>(),
            self.args
        )
    }
}
