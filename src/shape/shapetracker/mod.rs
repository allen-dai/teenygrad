pub mod util;
pub mod view;

use std::sync::{Arc, Mutex, MutexGuard};

pub use util::*;
use view::View;

use crate::shape::symbolic::{num, var};

use super::symbolic::ArcNode;

#[derive(Clone, Debug)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    pub fn new(shape: &[isize], views: Option<Vec<View>>) -> Self {
        let views = if let Some(v) = views {
            v
        } else {
            vec![View::new(shape, None, None, None)]
        };

        Self { views }
    }

    pub fn contiguous(&self) -> bool {
        self.views.len() == 1 && self.views[0].contiguous
    }

    pub fn shape(&self) -> Vec<isize> {
        self.views.last().unwrap().shape.clone()
    }
    pub fn strides(&self) -> Vec<isize> {
        self.views.last().unwrap().strides.clone()
    }

    pub fn key(&self) -> Vec<View> {
        self.views.clone()
    }

    pub fn size(&self) -> isize {
        let v = self.views.last().unwrap();
        v.shape
            .iter()
            .zip(v.strides.iter())
            .filter(|(_, &st)| st != 0)
            .map(|(sh, _)| *sh)
            .product()
    }

    pub fn real_offset(&self) -> isize {
        let (real_offset, mask) = self.expr_node(Some(var("zero", 0, 0)));
        assert!(real_offset.is_num());
        real_offset.num_val().unwrap()
    }

    pub fn real_strides(&self, ignore_valid: bool) -> Vec<Option<isize>> {
        let last_view = self.views.last().unwrap();
        if self.views.len() == 1 && last_view.mask.is_none() {
            return last_view.strides.iter().map(|st| Some(*st)).collect();
        };
        let mut ret = vec![None; last_view.shape.len()];
        let idxs: Vec<ArcNode> = self
            .shape()
            .iter()
            .enumerate()
            .map(|(i, sh)| var(&format!("idx{}", i), 0, sh - 1))
            .collect();
        let (idx, valid) = self.expr_idxs(Some(idxs.clone()));
        for this_dim in (if idx.is_sum() {
            idx.nodes()
        } else {
            vec![idx.clone()]
        }) {
            // println!("\n----\nidxs: {:?}\n\nidx: {}", idxs.iter().map(|n| n.key()).collect::<Vec<String>>(), this_dim.a().unwrap());
            if this_dim.is_mul()
                && this_dim.a().unwrap().is_var()
                && idxs.contains(&this_dim.a().unwrap())
            {
                ret[idxs
                    .iter()
                    .position(|n| n == &this_dim.a().unwrap())
                    .unwrap()] = Some(this_dim.b().unwrap().num_val().unwrap());
            } else if this_dim.is_var() {
                ret[idxs.iter().position(|n| n == &this_dim).unwrap()] = Some(1);
            }
        }
        let (idx_vars, valid_vars) = (idx.vars(), valid.vars());
        for (i, tidx) in idxs.iter().enumerate() {
            if valid_vars.contains(tidx) && !ignore_valid {
                ret[i] = None;
            } else if !idx_vars.contains(tidx) {
                ret[i] = Some(0);
            }
        }
        ret
    }

    pub fn _expr_idx(&self, mut idx: ArcNode, mut valid: ArcNode) -> (ArcNode, ArcNode) {
        for v in self.views[0..self.views.len() - 1].iter().rev() {
            if valid.max().unwrap() == 0 {
                return (num(-1), valid);
            }
            valid = v.expr_node_mask(idx.clone(), Some(valid));
            idx = v.expr_node(Some(idx));
        }
        return (idx, valid);
    }

    pub fn simplify(&mut self) {
        if self.views.len() < 2 {
            return;
        }
        let l = self.views.len();
        if let Some(new_view) = merge_view(&self.views[l - 2], &self.views[l - 1]) {
            self.views.pop();
            self.views.pop();
            self.views.push(new_view);
            self.simplify();
        }
    }

    pub fn expr_idxs(&self, idxs: Option<Vec<ArcNode>>) -> (ArcNode, ArcNode) {
        let idxs = if let Some(i) = idxs {
            i
        } else {
            self.shape()
                .iter()
                .enumerate()
                .map(|(i, sh)| var(&format!("idx{}", i), 0, sh - 1))
                .collect()
        };
        let idx = self.views[self.views.len() - 1].expr_idxs(&idxs);
        let valid = self.views[self.views.len() - 1].expr_node_mask(
            idxs_to_idx(&self.views[self.views.len() - 1].shape, &idxs),
            None,
        );
        self._expr_idx(idx, valid)
    }

    pub fn expr_node(&self, idx: Option<ArcNode>) -> (ArcNode, ArcNode) {
        let idx = if let Some(i) = idx {
            i
        } else {
            var("idx", 0, self.views.last().unwrap().shape.iter().product::<isize>() - 1)
        };
        self._expr_idx(
            self.views[self.views.len() - 1].expr_node(Some(idx.clone())),
            self.views[self.views.len() - 1].expr_node_mask(idx, None),
        )
    }

    pub fn expr_node_str(&self, idx: &str) -> (ArcNode, ArcNode) {
        let idx = var("idx", 0, self.shape().iter().product());
        self._expr_idx(
            self.views[self.views.len() - 1].expr_node(Some(idx.clone())),
            self.views[self.views.len() - 1].expr_node_mask(idx, None),
        )
    }

    pub fn axis_is_masked(&self, axis: isize) -> bool {
        let (_, valid) = self.expr_idxs(None);
        valid
            .vars()
            .iter()
            .any(|n| n.expr().is_some() && n.expr().unwrap() == &format!("idx{axis}"))
    }

    pub fn pad(&mut self, arg: &[(isize, isize)]) -> &mut Self {
        let last = self.views.last_mut().unwrap();
        *last = last.pad(arg);
        self
    }

    pub fn shrink(&mut self, arg: &[(isize, isize)]) -> &mut Self {
        let last = self.views.last_mut().unwrap();
        *last = last.shrink(arg);
        self
    }

    pub fn expand(&mut self, new_shape: &[isize]) -> &mut Self {
        let last = self.views.last_mut().unwrap();
        *last = last.expand(new_shape);
        self
    }

    pub fn reshape(&mut self, new_shape: &[isize]) -> &mut Self {
        let new_view = self.views[self.views.len() - 1].reshape(new_shape);
        if let Some(view) = new_view {
            *self.views.last_mut().unwrap() = view;
        } else {
            let extra = View::new(new_shape, None, None, None);
            //NOTE: Some werid cases when reshape [3,3,3,3] to [9,9], stride become [0,0] or [x,1]
            if let Some(merged_view) = merge_view(self.views.last().unwrap(), &extra) {
                *self.views.last_mut().unwrap() = merged_view;
            } else {
                self.views.push(extra);
            }
        }
        self
    }

    pub fn permute(&mut self, axis: &[isize]) -> &mut Self {
        let last = self.views.last_mut().unwrap();
        *last = last.permute(axis);
        self
    }

    pub fn stride(&mut self, mul: &[isize]) -> &mut Self {
        let last = self.views.last_mut().unwrap();
        *last = last.stride(mul);
        self
    }
}

#[test]
fn st_test() {
    let mut st = ShapeTracker::new(&[3, 3, 3], None);
    st.reshape(&[1, 3, 3, 3])
        .expand(&[3, 3, 3, 3])
        .reshape(&[9, 9]);
    println!("{:?}", st.views);
    println!("{:?} {:?}", st.shape(), st.strides());
    // st.permute(&[-2, 0, -1]);
    // println!("{:?}", st.strides());
}
