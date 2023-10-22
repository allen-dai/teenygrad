use std::{collections::HashMap, sync::Arc};

use crate::{
    codegen::linearizer::Args,
    lazy::LazyBuffer,
    runtime::Runtime,
    shape::{shapetracker::ShapeTracker, symbolic::Variable}, dtype::DType,
};

#[allow(unused_variables)]
pub trait Op: 'static + core::fmt::Debug + Send + Sync {
    // Unary
    fn neg(&self, x: &str) -> String {
        unimplemented!()
    }
    fn exp2(&self, x: &str) -> String {
        unimplemented!()
    }
    fn log2(&self, x: &str) -> String {
        unimplemented!()
    }
    fn sin(&self, x: &str) -> String {
        unimplemented!()
    }
    fn sqrt(&self, x: &str) -> String {
        unimplemented!()
    }

    // Binary
    fn add(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn sub(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn mul(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn div(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn _mod(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn bmax(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }
    fn cmplt(&self, a: &str, b: &str) -> String {
        unimplemented!()
    }

    // Reduce
    fn sum(&self, x: &str, shape: Vec<isize>) -> String {
        unimplemented!()
    }
    fn max(&self, x: &str, shape: Vec<isize>) -> String {
        unimplemented!()
    }

    // Ternary
    fn mulacc(&self, a: &str, b: &str, c: &str) -> String {
        unimplemented!()
    }
    fn _where(&self, a: &str, b: &str, c: &str) -> String {
        unimplemented!()
    }

    fn call(&self, op: &OpType, args: Vec<String>, shape: Option<Vec<isize>>) -> String {
        match op {
            OpType::Unary(uop) => {
                assert!(args.len() >= 1);
                match uop {
                    Unary::Neg => self.neg(&args[0]),
                    Unary::Exp2 => self.exp2(&args[0]),
                    Unary::Log2 => self.log2(&args[0]),
                    Unary::Sin => self.sin(&args[0]),
                    Unary::Sqrt => self.sqrt(&args[0]),
                    Unary::Noop => String::new(),
                }
            }
            OpType::Binary(bop) => {
                assert!(args.len() >= 2);
                match bop {
                    Binary::Add => self.add(&args[0], &args[1]),
                    Binary::Sub => self.sub(&args[0], &args[1]),
                    Binary::Mul => self.mul(&args[0], &args[1]),
                    Binary::Div => self.div(&args[0], &args[1]),
                    Binary::Mod => self._mod(&args[0], &args[1]),
                    Binary::Max => self.bmax(&args[0], &args[1]),
                    Binary::Cmplt => self.cmplt(&args[0], &args[1]),
                }
            }
            OpType::Reduce(rop) => {
                assert!(args.len() >= 1 && shape.is_some());
                match rop {
                    Reduce::Sum => self.sum(&args[0], shape.unwrap()),
                    Reduce::Max => self.max(&args[0], shape.unwrap()),
                }
            }
            OpType::Ternary(top) => {
                assert!(args.len() >= 3);
                match top {
                    Ternary::Mulacc => self.mulacc(&args[0], &args[1], &args[2]),
                    Ternary::Where => self._where(&args[0], &args[1], &args[2]),
                }
            }
            _ => unimplemented!("Does not need to implement the rest"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Unary {
    Neg,
    Exp2,
    Log2,
    Sin,
    Sqrt,
    Noop,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Max,
    Cmplt,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Reduce {
    Sum,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ternary {
    Mulacc,
    Where,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Movement {
    Reshape,
    Permute,
    Pad,
    Expand,
    Shrink,
    Stride,
    AsStrided
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Load {
    Empty,
    Rand,
    Const,
    From,
    Contiguous,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Buffer {
    Const,
    Mem
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    Unary(Unary),
    Binary(Binary),
    Reduce(Reduce),
    Ternary(Ternary),
    Movement(Movement),
    Load(Load),
    Buffer(Buffer)
}

macro_rules! optype_cmp {
    ($t: ident) => {
        impl PartialEq<$t> for OpType {
            fn eq(&self, other: &$t) -> bool {
                match self {
                    OpType::$t(x) => x == other,
                    _ => false,
                }
            }
        }
    };
}

optype_cmp!(Unary);
optype_cmp!(Binary);
optype_cmp!(Reduce);
optype_cmp!(Ternary);
optype_cmp!(Movement);
optype_cmp!(Load);
optype_cmp!(Buffer);

#[derive(Debug, Clone)]
pub enum LazyOpSrc {
    LazyOp(LazyOp),
    LazyBuffer(LazyBuffer),
}

impl From<LazyOp> for LazyOpSrc {
    fn from(value: LazyOp) -> Self {
        LazyOpSrc::LazyOp(value)
    }
}

impl From<LazyBuffer> for LazyOpSrc {
    fn from(value: LazyBuffer) -> Self {
        LazyOpSrc::LazyBuffer(value)
    }
}

impl LazyOpSrc {
    pub fn is_realized(&self) -> bool {
        match self {
            LazyOpSrc::LazyOp(op) => op.is_realized(),
            LazyOpSrc::LazyBuffer(lb) => lb.is_realized(),
        }
    }

    pub fn to_lb(self) -> LazyBuffer {
        match self {
            LazyOpSrc::LazyOp(_) => panic!("Lazyop cant turn into lazybuffer"),
            LazyOpSrc::LazyBuffer(lb) => lb,
        }
    }

    pub fn to_lo(self) -> LazyOp {
        match self {
            LazyOpSrc::LazyOp(lo) => lo,
            t => panic!("{t:?} cant turn into lazyop"),
        }
    }

    pub fn lb(&self) -> &LazyBuffer {
        match self {
            LazyOpSrc::LazyOp(_) => panic!("Lazyop cant turn into lazybuffer"),
            LazyOpSrc::LazyBuffer(lb) => lb,
        }
    }

    pub fn lo(&self) -> &LazyOp {
        match self {
            LazyOpSrc::LazyOp(lo) => lo,
            LazyOpSrc::LazyBuffer(_) => panic!("Lazyop cant turn into lazyop"),
        }
    }

    pub fn lb_mut(&mut self) -> &mut LazyBuffer {
        match self {
            LazyOpSrc::LazyOp(_) => panic!("Lazyop cant turn into lazybuffer"),
            LazyOpSrc::LazyBuffer(lb) => lb,
        }
    }

    pub fn lo_mut(&mut self) -> &mut LazyOp {
        match self {
            LazyOpSrc::LazyOp(lo) => lo,
            LazyOpSrc::LazyBuffer(_) => panic!("Lazyop cant turn into lazyop"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LazyOp {
    pub optype: OpType,
    pub src: Vec<LazyOpSrc>,
    pub buffers: Vec<LazyBuffer>,
    pub args: Vec<Args>,
}

pub trait LazyOpsDefaultImpl {
    fn st(&self) -> ShapeTracker {
        unimplemented!();
    }

    fn is_realized(&self) -> bool {
        unimplemented!();
    }

    fn children(&self) -> &[LazyBuffer] {
        unimplemented!();
    }
}

impl LazyOpsDefaultImpl for LazyOp {}

impl LazyOp {
    pub fn new<LOS: Into<LazyOpSrc>>(
        optype: OpType,
        src: Vec<LOS>,
        mut args: Option<Vec<Args>>,
    ) -> Self {
        let mut buffers = vec![];
        let src: Vec<LazyOpSrc> = src.into_iter().map(|s| s.into()).collect();
        for s in &src {
            match s {
                LazyOpSrc::LazyOp(x) => buffers.extend(x.buffers.clone()),
                LazyOpSrc::LazyBuffer(x) => buffers.push(x.clone()),
            }
        }
        let args = if args.is_none() {
            vec![]
        } else {
            args.take().unwrap()
        };
        Self {
            optype,
            src,
            buffers,
            args,
        }
    }

    pub fn map_buffers(&self, real_srcs: &HashMap<&LazyBuffer, Option<LazyOpSrc>>) -> Self {
        let mut srcs = vec![];
        for y in self.src.iter() {
            let ss = match y {
                LazyOpSrc::LazyOp(lop) => LazyOpSrc::LazyOp(lop.map_buffers(real_srcs)),
                LazyOpSrc::LazyBuffer(lb) => LazyOpSrc::LazyBuffer(lb.map_buffers(real_srcs)),
            };
            srcs.push(ss);
        }
        LazyOp::new(self.optype.clone(), srcs, Some(self.args.clone()))
    }

    pub fn get_lazyops(&self) -> Vec<LazyOp> {
        let mut ret = vec![self.clone()];
        for x in &self.src {
            if matches!(x, LazyOpSrc::LazyOp(_)) {
                ret.extend(x.lo().get_lazyops())
            }
        }
        ret
    }
}

#[derive(Debug, Clone)]
pub struct ScheduleItem {
    pub ast: LazyOp,
    pub out: LazyBuffer,
    pub inputs: Vec<LazyBuffer>,
    pub var_vals: HashMap<Variable, isize>,
}

#[derive(Debug, Clone)]
pub struct MemBuffer {
    pub idx: isize,
    pub dtype: DType,
    pub st: ShapeTracker
}


#[derive(Debug, Clone)]
pub struct ConstBuffer<T> {
    pub val: T,
    pub dtype: DType,
    pub st: ShapeTracker
}

#[derive(Clone)]
pub struct ASTRunner {
    name: String,
    prg: String,
    global_size: usize,
    local_size: usize,
    op_estimate: usize,
    mem_estimate: usize,
    display_name: String,
    args: Vec<String>,
    clprg: Arc<dyn Runtime>,
}

#[test]
fn somet() {
    let a = OpType::Binary(Binary::Add);
    println!("{:?}", a);
}
