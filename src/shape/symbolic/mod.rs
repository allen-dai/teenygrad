pub mod core_ops;

use dyn_clone::DynClone;
use std::{cmp::PartialEq, collections::HashMap, fmt::Display, rc::Rc, sync::Arc};

pub trait Node: core::fmt::Debug {
    // Node variants possible fields
    fn a(&self) -> Option<&dyn Node> {
        None
    }

    fn b(&self) -> Option<&dyn Node> {
        None
    }

    fn num_val(&self) -> Option<isize> {
        None
    }

    fn expr(&self) -> Option<&str> {
        None
    }

    fn key(&self) -> String {
        // @functools.cached_property
        // def key(self) -> str: return self.render(ctx="DEBUG")
        //
        //TODO: This i dont understand, render ops is None, so other backend render type just
        // fall back in to "render_python"????
        //        def render(self, ops=None, ctx=None, strip_parens=False) -> str:
        //            if ops is None: ops = render_python
        self.render(Box::new(CL {}), None, false)
    }

    fn min(&self) -> Option<isize> {
        None
    }

    fn max(&self) -> Option<isize> {
        None
    }

    fn _add(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
        sum(&[(*self)._clone(), rhs])
    }

    fn _sub(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
        self._add(rhs.neg())
    }

    fn _mul(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
        if let Some(b) = rhs.num_val() {
            if b == 0 {
                return num(0);
            }
            if b == 1 {
                return self._clone();
            }
        }
        if self.is_num() {
            if rhs.is_num() {
                return num(self.num_val().unwrap() * rhs.num_val().unwrap());
            } else {
                return rhs._mul(self.b().unwrap()._clone());
            }
        }
        MulNode::new(self._clone(), rhs.clone())
    }

    fn _div(&self, rhs: Box<dyn Node>, factoring_allowed: Option<bool>) -> Box<dyn Node> {
        if self.key() == rhs.key() {
            return num(1);
        }
        if (rhs._sub(self._clone())).min().unwrap() > 0 && self.min().unwrap() >= 0 {
            return num(0);
        }
        let b = rhs.num_val().unwrap();
        assert!(b != 0);
        if b < 0 {
            return self._div(num(-b), Some(true)) * -1;
        }
        if b == 1 {
            return self._clone();
        }
        let min = self.min().unwrap();
        if min < 0 {
            let offset = min / b;
            return (self._add(num(-offset)._mul(num(b)))._div(rhs, Some(false)))._add(num(offset));
        }
        DivNode::new((*self)._clone(), rhs)
    }

    fn neg(&self) -> Box<dyn Node> {
        self._mul(num(-1))
    }

    fn is_add(&self) -> bool {
        false
    }

    fn is_sub(&self) -> bool {
        false
    }

    fn is_mul(&self) -> bool {
        false
    }

    fn is_div(&self) -> bool {
        false
    }

    fn is_mod(&self) -> bool {
        false
    }

    fn is_and(&self) -> bool {
        false
    }

    fn is_num(&self) -> bool {
        false
    }

    fn is_sum(&self) -> bool {
        false
    }

    fn is_var(&self) -> bool {
        false
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String;

    fn vars(&self) -> Vec<&dyn Node> {
        vec![
            if let Some(a) = self.a() {
                a.vars()
            } else {
                vec![]
            },
            if let Some(b) = self.b() {
                b.vars()
            } else {
                vec![]
            },
            self.nodes(),
        ]
        .concat()
    }

    fn nodes(&self) -> Vec<&dyn Node> {
        vec![]
    }

    fn flat_components(&self) -> Vec<Box<dyn Node>> {
        vec![self._clone()]
    }

    fn _clone(&self) -> Box<dyn Node>;

    fn get_bounds(&self) -> Option<(isize, isize)> {
        None
    }
}

impl PartialEq for Box<dyn Node> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_num() && other.is_num() {
            return self.num_val().unwrap() == other.num_val().unwrap();
        }
        self.key() == other.key()
    }
}

impl Eq for Box<dyn Node> {}

//NOTE: Can remove this, _clone() and clone() is the same thing, it can be confusing.
// Having this is just nice when you need to do .to_vec() or .clone() a vec/slice of Box<dyn Node>
impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self._clone()
    }
}

impl std::hash::Hash for Box<dyn Node> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

pub fn var(expr: &str, min: isize, max: isize) -> Box<dyn Node> {
    Variable::new(expr, min, max)
}

pub fn num(v: isize) -> Box<dyn Node> {
    NumNode::new(v)
}

pub fn sum(_nodes: &[Box<dyn Node>]) -> Box<dyn Node> {
    let mut nodes = vec![];
    for n in _nodes {
        if n.min().is_some() || n.max().is_some() {
            nodes.push((*n)._clone())
        }
    }
    if nodes.len() == 0 {
        return num(0);
    }

    if nodes.len() == 1 {
        return nodes[0]._clone();
    }

    let mut new_nodes = vec![];
    let mut num_node_sum = 0;
    for n in (SumNode { nodes }).flat_components() {
        if n.is_num() {
            num_node_sum += n.num_val().unwrap();
            continue;
        }
        new_nodes.push(n)
    }
    let mut unique_mul_a: Vec<Box<dyn Node>> = new_nodes
        .clone()
        .into_iter()
        .filter(|n| n.is_mul())
        .map(|n| n.a().unwrap()._clone())
        .collect();
    unique_mul_a.dedup_by(|a, b| a.key() == b.key());
    if new_nodes.len() > 1 && unique_mul_a.len() < new_nodes.len() {
        new_nodes = factorize(new_nodes);
    }
    if num_node_sum > 0 {
        new_nodes.push(num(num_node_sum));
    }
    Box::new(SumNode { nodes: new_nodes })
}


// pub fn ands(nodes: &[Box<dyn Node>]) -> Box<dyn Node> {
//     AndNode::new(nodes)
// }


pub fn factorize(nodes: Vec<Box<dyn Node>>) -> Vec<Box<dyn Node>> {
    let mut mul_groups: HashMap<Box<dyn Node>, isize> = HashMap::new();
    for x in nodes {
        let (a, b) = if x.is_mul() {
            (x.a().unwrap()._clone(), x.b().unwrap()._clone())
        } else {
            (x, num(1))
        };
        *mul_groups
            .entry(a)
            .or_insert(b.num_val().expect("this should have a int but not sure")) +=
            b.num_val().unwrap();
    }
    let mut ret = vec![];
    for (a, b_sum) in mul_groups.keys().zip(mul_groups.values()) {
        if *b_sum == 0 {
            continue;
        }

        if *b_sum != 1 {
            ret.push(MulNode::new(a._clone(), num(*b_sum)));
        } else {
            ret.push(a._clone())
        }
    }
    ret
}

#[derive(Debug)]
pub struct Variable {
    expr: String,
    min: isize,
    max: isize,
}

impl Variable {
    fn new(expr: &str, min: isize, max: isize) -> Box<dyn Node> {
        Box::new(Self {
            expr: expr.to_string(),
            min,
            max,
        })
    }
}

impl Node for Variable {
    fn is_var(&self) -> bool {
        true
    }

    fn _clone(&self) -> Box<dyn Node> {
        Box::new(Self {
            expr: self.expr.clone(),
            min: self.min,
            max: self.max,
        })
    }

    fn expr(&self) -> Option<&str> {
        Some(&self.expr)
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        let mut ret = ops.variable(self._clone(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret.replace("(", "").replace(")", "");
        }
        ret
    }
}

#[derive(Debug)]
pub struct SumNode {
    nodes: Vec<Box<dyn Node>>,
}

impl SumNode {}

impl Node for SumNode {
    // Not sure about this, in sum() there is a check for min or max. but why. what node doesnt
    // have min max and is it safe to do this?
    fn min(&self) -> Option<isize> {
        Some(self.nodes.iter().map(|x| x.min().unwrap()).sum())
    }

    fn max(&self) -> Option<isize> {
        Some(self.nodes.iter().map(|x| x.max().unwrap()).sum())
    }

    fn _mul(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
        let mut a: Vec<Box<dyn Node>> = self.nodes.iter().map(|n| n._mul(rhs._clone())).collect();
        sum(&a)
    }

    fn is_sum(&self) -> bool {
        true
    }

    fn _clone(&self) -> Box<dyn Node> {
        Box::new(Self {
            nodes: self.nodes.iter().map(|x| (*x)._clone()).collect(),
        })
    }

    fn nodes(&self) -> Vec<&dyn Node> {
        self.nodes.iter().map(|x| x.as_ref()).collect()
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.sum(self._clone(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn flat_components(&self) -> Vec<Box<dyn Node>> {
        self.nodes
            .iter()
            .map(|n| n.flat_components())
            .collect::<Vec<Vec<Box<dyn Node>>>>()
            .concat()
    }
}

#[derive(Debug)]
pub struct MulNode {
    a: Box<dyn Node>,
    b: Box<dyn Node>,
    min: isize,
    max: isize,
}

impl MulNode {
    fn new(a: Box<dyn Node>, b: Box<dyn Node>) -> Box<dyn Node> {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };
        let (min, max) = ret
            .get_bounds()
            .expect("OpNode Mul should have impl get_bounds()");
        ret.min = min;
        ret.max = max;
        Box::new(ret)
    }
}

impl Node for MulNode {
    fn get_bounds(&self) -> Option<(isize, isize)> {
        let b = self.b.num_val().unwrap();
        if b >= 0 {
            return Some((self.a.min().unwrap() * b, self.a.max().unwrap() * b));
        }
        Some((self.a.max().unwrap() * b, self.a.min().unwrap() * b))
    }

    fn _mul(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
        self.a._mul((self.b._mul(rhs)))
    }

    // fn div(&self, rhs: Box<dyn Node>) -> Box<dyn Node> {
    //
    // }

    fn is_mul(&self) -> bool {
        true
    }
    fn _clone(&self) -> Box<dyn Node> {
        Box::new(Self {
            a: self.a._clone(),
            b: self.b._clone(),
            min: self.min,
            max: self.max,
        })
    }

    fn a(&self) -> Option<&dyn Node> {
        Some(self.a.as_ref())
    }

    fn b(&self) -> Option<&dyn Node> {
        Some(self.b.as_ref())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }
    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.mul(self._clone(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret.replace("(", "").replace(")", "");
        }
        ret
    }
}

#[derive(Debug)]
pub struct NumNode {
    b: isize,
}

impl NumNode {
    pub fn new(b: isize) -> Box<dyn Node> {
        Box::new(Self { b })
    }
}

impl Node for NumNode {
    fn num_val(&self) -> Option<isize> {
        Some(self.b)
    }

    fn b(&self) -> Option<&dyn Node> {
        Some(self)
    }

    fn min(&self) -> Option<isize> {
        Some(self.b)
    }

    fn max(&self) -> Option<isize> {
        Some(self.b)
    }

    fn is_num(&self) -> bool {
        true
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        self.b.to_string()
    }

    fn _clone(&self) -> Box<dyn Node> {
        Box::new(Self { b: self.b })
    }
}

#[derive(Debug)]
pub struct DivNode {
    a: Box<dyn Node>,
    b: Box<dyn Node>,
    min: isize,
    max: isize,
}

impl DivNode {
    fn new(a: Box<dyn Node>, b: Box<dyn Node>) -> Box<dyn Node> {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };
        let (min, max) = ret
            .get_bounds()
            .expect("OpNode Div should have impl get_bounds()");
        ret.min = min;
        ret.max = max;
        Box::new(ret)
    }
}

impl Node for DivNode {
    fn _div(&self, rhs: Box<dyn Node>, factoring_allowed: Option<bool>) -> Box<dyn Node> {
        self.a._div(self.b._mul(rhs), None)
    }

    fn get_bounds(&self) -> Option<(isize, isize)> {
        Some((
            self.a.min().unwrap() / self.b.num_val().unwrap(),
            self.a.max().unwrap() / self.b.num_val().unwrap(),
        ))
    }

    fn is_div(&self) -> bool {
        true
    }

    fn _clone(&self) -> Box<dyn Node> {
        Box::new(Self {
            a: self.a._clone(),
            b: self.b._clone(),
            min: self.min,
            max: self.max,
        })
    }

    fn a(&self) -> Option<&dyn Node> {
        Some(self.a.as_ref())
    }

    fn b(&self) -> Option<&dyn Node> {
        Some(self.b.as_ref())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Box<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.div(self._clone(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret.replace("(", "").replace(")", "");
        }
        ret
    }
}

pub trait NodeOp {
    fn variable(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        // Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
        if ctx.is_some_and(|f| f == "DEBUG") {
            return format!(
                "{}[{}-{}]",
                s.expr().unwrap(),
                s.min().unwrap(),
                s.max().unwrap()
            );
        }
        s.expr().unwrap().to_string()
    }

    fn num(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        // NumNode: lambda self,ops,ctx: f"{self.b}",
        s.b().unwrap().to_string()
    }

    fn mul(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        // MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{sym_render(self.b,ops,ctx)})",
        format!(
            "({}*{})",
            s.a().unwrap().render(Box::new(CL), ctx, false),
            s.b().unwrap().render(Box::new(CL), ctx, false), // <-- Everything should be a Node here,
                                                               // so no need to "sym_render()"
        )
    }

    fn div(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        // DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}/{self.b})",
        format!(
            "({}/{})",
            s.a().unwrap().render(Box::new(CL), ctx, false),
            s.b().unwrap()
        )
    }

    fn _mod(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        // ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
        format!(
            "({}%{})",
            s.a().unwrap().render(Box::new(CL), ctx, false),
            s.b().unwrap()
        )
    }

    fn lt(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        //LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
        format!(
            "({}<{})",
            s.a().unwrap().render(Box::new(CL), ctx, false),
            s.b().unwrap().render(Box::new(CL), ctx, false),
        )
    }

    fn sum(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        let mut renders = vec![];
        for n in s.nodes() {
            renders.push(n.render(Box::new(CL), ctx, false));
        }
        renders.sort();
        format!("({})", renders.join("+"))
    }

    fn and(&self, s: Box<dyn Node>, ctx: Option<&str>) -> String {
        let mut renders = vec![];
        for n in s.nodes() {
            renders.push(n.render(Box::new(CL), ctx, false));
        }
        renders.sort();
        format!("({})", renders.join("&&"))
    }
}

pub struct CL;
impl NodeOp for CL{}

#[test]
fn sym_test() {
    let a = var("x", 0, 1800);
    let b = a + 10 + 9 * 10 / 10;
    println!("{b}");
}
