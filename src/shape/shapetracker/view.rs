use super::util::*;
use crate::shape::symbolic::{ands, num, sum, var, ArcNode};

#[macro_export]
macro_rules! view {
    ($shape: expr) => {
        View::new(&$shape.to_vec(), None, None, None)
    };
    ($shape: expr, $strides: expr) => {
        View::new(&$shape.to_vec(), Some($strides.to_vec()), None, None)
    };
    ($shape: expr, $strides: expr, $offset: expr) => {
        View::new(
            &$shape.to_vec(),
            Some($strides.to_vec()),
            Some($offset),
            None,
        )
    };
    ($shape: expr, $strides: expr, $offset: expr, $mask: expr) => {
        View::new(
            &$shape.to_vec(),
            Some($strides.to_vec()),
            Some($offset),
            Some($mask.to_vec()),
        )
    };
}

#[derive(Debug, Clone)]
pub struct View {
    pub shape: Vec<isize>,
    pub strides: Vec<isize>,
    pub offset: isize,
    pub mask: Option<Vec<(isize, isize)>>,
    pub shape_strides: Vec<(isize, isize)>,
    pub contiguous: bool,
}

impl View {
    pub fn new(
        shape: &[isize],
        strides: Option<Vec<isize>>,
        offset: Option<isize>,
        mask: Option<Vec<(isize, isize)>>,
    ) -> Self {
        let strides = if let Some(s) = strides {
            filter_strides(&shape, &s)
        } else {
            strides_for_shape(&shape)
        };
        let offset = if let Some(val) = offset { val } else { 0 };
        let contiguous = offset == 0
            && mask.is_none()
            && strides
                .iter()
                .zip(strides_for_shape(shape).iter())
                .all(|(s1, s2)| s1 == s2);
        Self {
            shape_strides: to_shape_strides(&shape, &strides),
            shape: shape.to_vec(),
            strides,
            offset,
            mask,
            contiguous,
        }
    }

    pub fn expr_node_mask(&self, idx: ArcNode, valid: Option<ArcNode>) -> ArcNode {
        let mut expr = if let Some(n) = valid { vec![n] } else { vec![] };
        if let Some(mask) = &self.mask {
            let mut acc = 1;
            for (&ns, &(x, y)) in self.shape.iter().zip(mask).rev() {
                if x != 0 || y != ns {
                    let base = (&idx / acc) % ns;
                    expr.extend([base.ge(num(x)), base.lt(num(y))]);
                }
                acc *= ns;
            }
        }
        ands(&expr)
    }

    pub fn expr_node(&self, idx: Option<ArcNode>) -> ArcNode {
        let idx = if let Some(i) = idx {
            i
        } else {
            var("idx", 0, self.shape.iter().product::<isize>() - 1)
        };
        let mut ret = vec![];
        if self.offset != 0 {
            ret.push(num(self.offset));
        }
        let mut acc = 1;
        for &(d, s) in to_shape_strides(&self.shape, &self.strides).iter().rev() {
            ret.push(((&idx / acc) % d) * s);
            acc *= d;
        }
        sum(&ret)
    }

    pub fn expr_idxs(&self, idxs: &[ArcNode]) -> ArcNode {
        assert!(idxs.len() == self.shape.len());
        let mut ret = vec![];
        if self.offset != 0 {
            ret.push(num(self.offset));
        }
        for (idx, (&sh, &st)) in idxs.iter().zip(self.shape.iter().zip(self.strides.iter())) {
            if sh == 1 || st == 0 {
                continue;
            }
            ret.push(idx * st)
        }
        sum(&ret)
    }

    fn __unsafe_resize(
        &self,
        arg: &[(isize, isize)],
        mut mask: Option<Vec<(isize, isize)>>,
    ) -> Self {
        let offset = get_unsafe_resize_offset(&self.strides, &arg);
        if let Some(m) = &self.mask {
            let nmask: Vec<(isize, isize)> = m
                .iter()
                .zip(arg.iter())
                .map(|(&(mx, my), &(ax, ay))| ((mx - ax).max(0), (my - ax).min(ay - ax)))
                .collect();
            if mask.is_none() {
                mask = Some(nmask);
            } else {
                mask = Some(
                    nmask
                        .iter()
                        .zip(mask.unwrap().iter())
                        .map(|(&(mx1, my1), &(mx2, my2))| (mx1.max(mx2), my1.max(my2)))
                        .collect::<Vec<(isize, isize)>>(),
                );
            }
        }
        View::new(
            &arg.iter().map(|(x, y)| y - x).collect::<Vec<isize>>(),
            Some(self.strides.clone()),
            Some(self.offset + offset),
            mask,
        )
    }

    pub fn pad(&self, arg: &[(isize, isize)]) -> Self {
        assert!(arg.iter().all(|&(b, e)| b >= 0 && e >= 0) && arg.len() == self.shape.len());
        if arg.iter().all(|&(b, e)| b == 0 && e == 0) {
            return self.clone();
        }
        let (zvarg, mask) = get_pad_args(&self.shape, arg);
        self.__unsafe_resize(&zvarg, Some(mask))
    }

    pub fn shrink(&self, arg: &[(isize, isize)]) -> Self {
        assert!(
            self.shape
                .iter()
                .zip(arg.iter())
                .all(|(&sh, &(b, e))| b >= 0 && e <= sh)
                && arg.len() == self.shape.len()
        );
        self.__unsafe_resize(arg, None)
    }

    pub fn expand(&self, new_shape: &[isize]) -> Self {
        assert!(new_shape.len() == self.shape.len());
        assert!(self
            .shape
            .iter()
            .zip(new_shape.iter().zip(self.strides.iter()))
            .all(|(&s, (&x, &st))| s == x || (s == 1 && st == 0)));
        let mask = if let Some(m) = &self.mask {
            Some(
                m.iter()
                    .zip(self.shape.iter().zip(new_shape))
                    .map(|(&m, (&s, &ns))| {
                        if m != (0, 1) {
                            (0, 0)
                        } else {
                            if s != ns {
                                (0, ns)
                            } else {
                                m
                            }
                        }
                    })
                    .collect::<Vec<(isize, isize)>>(),
            )
        } else {
            None
        };
        View::new(
            new_shape,
            Some(self.strides.clone()),
            Some(self.offset),
            mask,
        )
    }

    pub fn reshape(&self, new_shape: &[isize]) -> Option<Self> {
        if self.shape == new_shape {
            return Some(self.clone());
        }
        assert!(new_shape.iter().all(|&sh| sh > 0));
        assert!(self.shape.iter().product::<isize>() == new_shape.iter().product::<isize>());
        if self.contiguous && self.shape.len() > new_shape.len() {
            return Some(View::new(new_shape, None, None, None));
        }

        if self
            .shape
            .iter()
            .filter(|&&x| x != 1)
            .eq(new_shape.iter().filter(|&&x| x != 1))
        {
            let mut new_strides: Vec<isize> = self
                .shape
                .iter()
                .zip(self.strides.iter())
                .filter(|(&x, _)| x != 1)
                .rev()
                .map(|(_, &y)| y)
                .collect();
            let new_strides_tuple: Vec<isize> = new_shape
                .iter()
                .map(|&x| {
                    if x == 1 {
                        0
                    } else {
                        new_strides.pop().unwrap()
                    }
                })
                .collect();
            let mut new_mask_tuple = None;
            if let Some(m) = &self.mask {
                for (&x, &y) in self.shape.iter().zip(m.iter()) {
                    if x == 1 && y != (0, 1) {
                        new_mask_tuple = Some(vec![(0, 0); new_shape.len()]);
                        break;
                    }
                }
                if new_mask_tuple.is_none() {
                    let mut new_mask: Vec<(isize, isize)> = self
                        .shape
                        .iter()
                        .zip(m.iter())
                        .filter(|(&sh, _)| sh != 1)
                        .map(|(_, &mm)| mm)
                        .rev()
                        .collect();
                    new_mask_tuple = Some(
                        new_shape
                            .iter()
                            .map(|&x| {
                                if x == 1 {
                                    (0, 1)
                                } else {
                                    new_mask.pop().unwrap()
                                }
                            })
                            .collect::<Vec<(isize, isize)>>(),
                    );
                }
            }
            return Some(View::new(
                new_shape,
                Some(new_strides_tuple),
                Some(self.offset),
                new_mask_tuple,
            ));
        }
        None
    }

    pub fn permute(&self, axis: &[isize]) -> Self {
        let mut new_shape = vec![];
        let mut new_stride = vec![];
        let mut new_mask = vec![];
        for &i in axis.iter() {
            let i = if i >= 0 {
                i as usize
            } else {
                (self.strides.len() as isize + i) as usize
            };
            new_shape.push(self.shape[i]);
            new_stride.push(self.strides[i]);
            if let Some(m) = &self.mask {
                new_mask.push(m[i])
            }
        }
        let new_view = View::new(
            &new_shape,
            Some(new_stride),
            Some(self.offset),
            if self.mask.is_some() {
                Some(new_mask)
            } else {
                None
            },
        );
        new_view
    }

    pub fn stride(&self, mul: &[isize]) -> Self {
        let strides: Vec<isize> = self
            .strides
            .iter()
            .zip(mul.iter())
            .map(|(z, m)| z * m)
            .collect();
        let new_shape: Vec<isize> = self
            .shape
            .iter()
            .zip(mul.iter())
            .map(|(s, m)| (s + (m.abs() - 1)) / m.abs())
            .collect();
        let offset: isize = self
            .shape
            .iter()
            .zip(self.strides.iter().zip(mul.iter()))
            .filter(|(_, (_, &m))| m < 0)
            .map(|(&s, (&z, _))| (s - 1) * z)
            .sum();
        //  tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(self.views[-1].mask, self.views[-1].shape, mul)]) if self.views[-1].mask is not None else None
        let mask = if let Some(m) = &self.mask {
            Some(
                m.iter()
                    .zip(self.shape.iter().zip(mul.iter()))
                    .map(|(&(mx, my), (&s, &m))| {
                        (
                            (if m > 0 { mx } else { s - my } + m.abs() - 1) / m.abs(),
                            (if m > 0 { my } else { s - mx } + m.abs() - 1) / m.abs(),
                        )
                    })
                    .collect::<Vec<(isize, isize)>>(),
            )
        } else {
            None
        };
        View::new(&new_shape, Some(strides), Some(offset), mask)
    }
}
