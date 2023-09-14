use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    backend::Backend,
    codegen::linearizer::Args,
    dtype::DType,
    ops::{LazyOp, LazyOpSrc, LazyOpsDefaultImpl, Load, Movement, Op, OpType},
    runtime::RawBuffer,
    shape::{shapetracker::ShapeTracker, symbolic::Variable},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct LazyBufferId(pub(crate) usize);

pub(crate) fn lb_id() -> LazyBufferId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    LazyBufferId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[derive(Clone)]
pub struct LazyBuffer {
    pub lazyop: LazyOp,
    pub st: ShapeTracker,
    pub _realized: Option<RawBuffer>,
    pub base: Option<Box<LazyBuffer>>,
    pub shape: Vec<isize>,
    pub children: HashSet<LazyBuffer>,
    pub views: HashSet<LazyBuffer>,
    pub id: LazyBufferId,
    pub dtype: DType,
    pub device: String,
}

impl PartialEq for LazyBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for LazyBuffer {}
impl std::hash::Hash for LazyBuffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl core::fmt::Debug for LazyBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<LB {:?} {:?} op={:?} st={:?}",
            self.shape, self.dtype, self.lazyop, self.st
        )
    }
}

impl LazyBuffer {
    pub fn new(
        device: &str,
        st: ShapeTracker,
        optype: OpType,
        mut op: LazyOp,
        dtype: DType,
        src: Option<RawBuffer>,
        mut base: Option<LazyBuffer>,
    ) -> Self {
        let base = {
            if base.is_none() {
                None
            } else {
                Some(Box::new(base.take().unwrap()))
            }
        };
        let mut ret = Self {
            device: device.into(),
            lazyop: op,
            shape: st.shape(),
            st,
            _realized: src,
            base,
            children: HashSet::new(),
            views: HashSet::new(),
            id: lb_id(),
            dtype,
        };
        let rc = ret.clone();
        for x in ret.lazyop.buffers.iter_mut() {
            x.children.insert(rc.clone());
        }
        if ret.base.is_some() {
            ret.base.as_mut().unwrap().views.insert(rc.clone());
        } else {
            assert!(ret.st.contiguous());
        }
        ret
    }

    pub fn is_realized(&self) -> bool {
        self.base.as_ref().is_some_and(|b| b._realized.is_some())
    }

    pub fn loadop(
        optype: OpType,
        shape: &[isize],
        dtype: DType,
        device: &str,
        args: Option<Vec<Args>>,
        src: Option<LazyBuffer>,
    ) -> Self {
        let mut ss = vec![];
        if let Some(src) = src {
            ss.push(src);
        };
        create_lazybuffer(
            device,
            ShapeTracker::new(shape, None),
            optype.clone(),
            LazyOp::new(optype, ss, args),
            dtype,
            None,
        )
    }

    pub fn const_(&self, val: String) -> Self {
        Self::loadop(
            OpType::Load(Load::Const),
            &vec![],
            self.dtype.clone(),
            &self.device,
            Some(vec![Args::Str(val)]),
            None,
        )
        .reshape(&vec![1; self.shape.len()])
        .expand(&self.shape)
    }

    pub fn _movement_op(&mut self, st: ShapeTracker, optype: OpType, arg: &[isize]) -> Self {
        if matches!(self.lazyop.optype, OpType::Binary(_))
            && !self.is_realized()
            && (matches!(
                optype,
                OpType::Movement(Movement::Shrink)
                    | OpType::Movement(Movement::Stride)
                    | OpType::Movement(Movement::Permute)
            ) || (matches!(optype, OpType::Movement(Movement::Reshape))
                && matches!(self.lazyop.optype, OpType::Unary(_))))
            && self.children.is_empty()
        {
            return match optype {
                OpType::Movement(m) => match m {
                    Movement::Reshape => self.reshape(arg),
                    Movement::Permute => todo!(),
                    Movement::Pad => todo!(),
                    Movement::Expand => self.expand(arg),
                    Movement::Shrink => todo!(),
                    Movement::Stride => todo!(),
                },
                _ => unreachable!(),
            };
        }
        if !self.is_realized() && st.contiguous() {
            let root = get_movementroot(&*self, false);
            if root.st.contiguous()
                && root != self
                && st.shape().iter().product::<isize>() == root.shape.iter().product::<isize>()
            {
                return root.clone().reshape(&st.shape());
            }
        }

        create_lazybuffer(
            &self.device,
            st,
            optype.clone(),
            LazyOp::new(
                optype,
                vec![self.clone()],
                Some(arg.iter().map(|i| Args::Int(*i)).collect::<Vec<Args>>()),
            ),
            self.dtype.clone(),
            None,
        )
    }

    pub fn reshape(&mut self, arg: &[isize]) -> Self {
        if self.shape == arg {
            return self.clone();
        }
        if !self.is_realized() && matches!(self.lazyop.optype, OpType::Movement(Movement::Reshape))
        {
            let s_clone = self.clone();
            self.lazyop.src[0].lb_mut().children.remove(&s_clone);
            return self.lazyop.src[0].lb_mut().reshape(arg);
        }
        self.st.reshape(arg);
        self._movement_op(self.st.clone(), OpType::Movement(Movement::Reshape), arg)
    }

    pub fn pad(&mut self, arg: &[isize]) -> Self {
        if arg.iter().all(|v| *v == 0) {
            return self.clone();
        }
        if !self.is_realized() && matches!(self.lazyop.optype, OpType::Movement(Movement::Pad)) {
            let op_arg = self
                .lazyop
                .args
                .iter()
                .map(|v| v.to_int())
                .collect::<Vec<isize>>();
            return self.lazyop.src[0].lb_mut().pad(&op_arg);
        }

        let mut aarg = vec![];
        for a in arg.windows(2) {
            aarg.push((a[0], a[1]))
        }
        self.st.pad(&aarg);
        self._movement_op(self.st.clone(), OpType::Movement(Movement::Pad), arg)
    }

    pub fn expand(&mut self, arg: &[isize]) -> Self {
        if &self.shape == arg {
            return self.clone();
        }
        if !self.is_realized() && matches!(self.lazyop.optype, OpType::Movement(Movement::Expand)) {
            return self.lazyop.src[0].lb_mut().expand(arg);
        }
        self.st.expand(arg);
        return self._movement_op(self.st.clone(), OpType::Movement(Movement::Expand), arg);
    }

    pub fn permute(&mut self, arg: &[isize]) -> Self {
        if arg == &(0..arg.len()).map(|v| v as isize).collect::<Vec<isize>>() {
            return self.clone();
        }
        if !self.is_realized() {
            return match &self.lazyop.optype {
                OpType::Movement(m) => match m {
                    Movement::Permute => self.lazyop.src[0].lb_mut().permute(
                        &self
                            .lazyop
                            .args
                            .iter()
                            .map(|v| v.to_int())
                            .collect::<Vec<isize>>(),
                    ),
                    Movement::Expand => self.lazyop.src[0].lb_mut().permute(arg).expand(
                        &arg.iter()
                            .map(|i| self.lazyop.args[*i as usize].to_int())
                            .collect::<Vec<isize>>(),
                    ),
                    _ => unreachable!(),
                },
                OpType::Reduce(_) => {
                    let narg = arg
                        .iter()
                        .map(|i| self.lazyop.args[*i as usize].clone())
                        .collect::<Vec<Args>>();
                    let s_clone = self.clone();
                    let src = self.lazyop.src[0].lb_mut();
                    let optype = &self.lazyop.optype;
                    src.children.remove(&s_clone);
                    //return src.permute(arg).r(cast(ReduceOps, rop), narg)
                    todo!()
                }
                _ => unreachable!(),
            };
        }
        self.st.permute(arg);
        self._movement_op(self.st.clone(), OpType::Movement(Movement::Permute), arg)
    }
}

pub fn create_lazybuffer(
    device: &str,
    st: ShapeTracker,
    optype: OpType,
    op: LazyOp,
    dtype: DType,
    base: Option<LazyBuffer>,
) -> LazyBuffer {
    if matches!(
        optype,
        OpType::Load(Load::Empty) | OpType::Load(Load::Rand) | OpType::Load(Load::Const)
    ) {
        return LazyBuffer::new(device, st, optype, op, dtype, None, base);
    }
    // # wop is the deduping key. i feel this used to compare more deeply
    // wop = (device, dtype, optype, ref(op), ref(base) if base else None)
    // if wop in lazycache:
    //   for x in op.buffers: x.children.add(lazycache[wop])
    //   return lazycache[wop]
    //
    // lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype, base=base)
    LazyBuffer::new(device, st, optype, op, dtype, None, base)
}

fn _ast_reduceops(op: &LazyOp) -> LazyOp {
    let mut ret = op.clone();
    if !op.src[0].is_realized() {
        if let LazyOpSrc::LazyOp(lop) = &op.src[0] {
            if matches!(op.optype, OpType::Binary(_)) && op.children().len() <= 1 {
                ret.src = vec![op.src[0].clone()];
            }
        } else {
            panic!("if not src.realized, then src.op must be a LazyOp")
        }
    }
    ret
}

fn _ast_binaryops(op: &LazyOp, shape: &[isize]) -> LazyOp {
    let mut real_srcs: HashMap<&LazyBuffer, Option<LazyOpSrc>> = {
        let mut m = HashMap::new();
        for x in &op.buffers {
            m.insert(x, None);
        }
        m
    };
    //[(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype == ReduceOps and not x.realized and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
    let mut psrcs: Vec<(&LazyBuffer, &LazyBuffer)> = real_srcs
        .keys()
        .into_iter()
        .map(|&v| (v, get_movementroot_contiguous(v)))
        .into_iter()
        .filter(|(k, x)| {
            matches!(x.lazyop.optype, OpType::Reduce(_))
                && !x.is_realized()
                && k.shape.iter().product::<isize>() == x.shape.iter().product::<isize>()
                && x.children.len() <= 1
                && k.children.len() <= 1
        })
        .collect();
    let mut intermediate_shape = shape;
    todo!()
}

fn get_single_root(root: &LazyBuffer) -> &LazyBuffer {
    if root.lazyop.src.len() == 1 && matches!(root.lazyop.src[0], LazyOpSrc::LazyBuffer(_)) {
        return get_single_root(root.lazyop.src[0].lb());
    };
    root
}

fn get_movementroot(root: &LazyBuffer, allow_contiguous: bool) -> &LazyBuffer {
    if !root.is_realized()
        && (matches!(root.lazyop.optype, OpType::Movement(_))
            || (matches!(root.lazyop.optype, OpType::Load(Load::Contiguous))
                && allow_contiguous
                && root.lazyop.src[0].lb().st.contiguous()))
    {
        return get_movementroot(root.lazyop.src[0].lb(), allow_contiguous);
    }
    root
}

fn get_movementroot_contiguous(x: &LazyBuffer) -> &LazyBuffer {
    if !x.is_realized() && matches!(x.lazyop.optype, OpType::Load(Load::Contiguous)) {
        return get_movementroot_contiguous(x.lazyop.src[0].lb());
    }
    if matches!(x.lazyop.optype, OpType::Movement(_)) && x.st.contiguous() {
        return get_movementroot(x, true);
    }
    x
}
