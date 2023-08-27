pub mod core_ops;
pub mod dtype;
pub mod shape;

use crate::backend::Backend;
use crate::prelude::*;
use rand_distr::StandardNormal;

pub struct Tensor<B: Backend> {
    pub(crate) inner: B,
    pub(crate) grad: Option<Vec<Tensor<B>>>,
}

impl<B: Backend> Tensor<B> {
    pub fn device(&self) -> String {
        std::any::type_name::<B>()
            .to_string()
            .split("::")
            .last()
            .unwrap()
            .to_string()
    }

    pub fn dtype(&self) -> String {
        std::any::type_name::<B::Dtype>().to_string()
    }

    pub fn from_vec<V: Into<Vec<B::Dtype>>, S: Into<Shape>>(data: V, shape: S) -> Self {
        let shape = shape.into();
        let data = data.into();
        assert!(data.len() == shape.numel());
        Self {
            inner: B::from_vec(data, &shape),
            grad: None,
        }
    }

    pub fn randn<S: Into<Shape>>(shape: S) -> Self
    where
        StandardNormal: rand_distr::Distribution<B::Dtype>,
    {
        Self {
            inner: B::rand(&shape.into(), StandardNormal),
            grad: None,
        }
    }

    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        use num_traits::Zero;
        Self {
            inner: B::empty(&shape.into()).const_like(B::Dtype::zero()),
            grad: None,
        }
    }

    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        use num_traits::One;
        Self {
            inner: B::empty(&shape.into()).const_like(B::Dtype::one()),
            grad: None,
        }
    }

    pub fn sum(&self, axis: isize) -> Self {
        Self {
            inner: B::sum(&self.inner, Some(axis), false),
            grad: None,
        }
    }

    pub fn sum_keepdim(&self, axis: isize) -> Self {
        Self {
            inner: B::sum(&self.inner, Some(axis), true),
            grad: None,
        }
    }

    pub fn sum_all(&self) -> Self {
        Self {
            inner: B::sum(&self.inner, None, false),
            grad: None,
        }
    }

    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::reshape(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn permute<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::permute(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Self {
        Self {
            inner: B::expand(&self.inner, shape.into()),
            grad: None,
        }
    }

    pub fn to_vec(&self) -> Vec<B::Dtype> {
        self.inner.to_vec()
    }

    pub fn shape(&self) -> Shape {
        B::shape(&self.inner)
    }

    pub fn stride(&self) -> Shape {
        B::stride(&self.inner)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        let n1 = self.shape().len();
        let n2 = rhs.shape().len();
        let mut a_shape = self.shape();
        let last = a_shape.dims.pop().unwrap();
        a_shape
            .dims
            .extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        a_shape.dims.push(last);
        let mut a = self.reshape(a_shape);
        let mut b_shape = rhs.shape();
        b_shape.dims = b_shape.dims[0..b_shape.len() - 2].to_vec();
        b_shape
            .dims
            .extend_from_slice(&vec![1; (n1 - 1).min(n2 - 1).min(1)]);
        b_shape
            .dims
            .extend_from_slice(&rhs.shape().dims[rhs.shape().len() - n2.min(2)..]);
        let mut order = (0..b_shape.len()).collect::<Vec<usize>>();
        order.swap(b_shape.len() - 1, b_shape.len() - 2.min(n2));
        let mut b = rhs.reshape(b_shape).permute(order);
        let (a, b) = Self::_broadcast(a, b);
        (a * b).sum(-1)
    }

    pub fn _broadcast(mut x: Self, mut y: Self) -> (Self, Self) {
        let mut xshape = x.shape();
        let mut yshape = y.shape();
        if xshape == yshape {
            return (x, y);
        }
        let shape_delta = xshape.len() - yshape.len();
        if shape_delta > 0 {
            let mut ysh = vec![1; shape_delta];
            ysh.extend_from_slice(&yshape.dims);
            y = y.reshape(ysh);
        }
        yshape = y.shape();
        if xshape == yshape {
            return (x, y);
        }

        let shape_ret = Shape::from(
            xshape
                .dims
                .iter()
                .zip(yshape.dims.iter())
                .map(|(x, y)| *x.max(y))
                .collect::<Vec<usize>>(),
        );
        if xshape != shape_ret {
            x = x.expand(shape_ret.clone());
        }
        if yshape != shape_ret {
            y = y.expand(shape_ret.clone());
        }
        (x, y)
    }

    pub fn transpose(&self, d1: isize, d2: isize) -> Self {
        let d1 = if d1 < 0 {
            (self.shape().len() as isize + d1) as usize
        } else {
            d1 as usize
        };
        let d2 = if d2 < 0 {
            (self.shape().len() as isize + d2) as usize
        } else {
            d2 as usize
        };

        let mut p = (0..self.shape().len()).collect::<Vec<usize>>();
        p.swap(d1, d2);
        self.permute(p)
    }

    pub fn shrink<A: Into<Vec<(usize, usize)>>>(&self, arg: A) -> Self {
        Tensor {
            inner: self.inner.shrink(arg),
            grad: None,
        }
    }

    pub fn pad<A: Into<Vec<(usize, usize)>>>(&self, arg: A, const_value: B::Dtype) -> Self {
        Tensor {
            inner: self.inner.pad(arg, const_value),
            grad: None,
        }
    }

    pub fn pad2d<A: Into<Vec<usize>>>(&self, padding: A, const_value: B::Dtype) -> Self {
        let padding = padding.into();
        let slc: Vec<(isize, isize)> = padding
            .iter()
            .step_by(2)
            .zip(
                padding[1..]
                    .iter()
                    .step_by(2)
                    .zip(self.shape().dims.iter().rev()),
            )
            .map(|(p0, (p1, s))| (-(*p0 as isize), *s as isize + *p1 as isize))
            .rev()
            .collect();
        let mut slice_shape: Vec<(isize, isize)> = self.shape().dims
            [..self.shape().len() - padding.len() / 2]
            .iter()
            .map(|sh| (0, *sh as isize))
            .collect();
        slice_shape.extend(slc.iter());
        self.slice(slice_shape, const_value)
    }

    #[rustfmt::skip]
    pub fn _pool<S: Into<Shape>>(&self, k_: S, stride: usize, dilation: usize) -> Self {
        let self_shape = self.shape();
        let k_ = k_.into();
        let d_ = Shape::from(vec![dilation;k_.len()]);
        let s_ = Shape::from(vec![stride;k_.len()]);
        assert!(self_shape.len() >= k_.len(), "can't pool {self_shape:?} with {k_:?}");
        assert!(k_.len() == s_.len() && s_.len() == d_.len(), "stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}");
        let slc_prefix: Vec<(usize, usize)> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| (0, *sh))
            .collect();
        let prefix: Vec<usize> = self_shape.dims[0..self_shape.len() - k_.len()]
            .iter()
            .map(|sh| *sh)
            .collect();
        let i_: Vec<usize> = self_shape.dims[self_shape.len() - k_.len()..]
            .iter()
            .map(|sh| *sh)
            .collect();

        let mut xup = if k_.dims.iter().zip(s_.dims.iter()).any(|(k, s)| k > s) || d_.dims.iter().any(|d| *d != 1)
        {
            let o_: Vec<usize> = i_
                .iter()
                .zip(d_.dims.iter().zip(k_.dims.iter().zip(s_.dims.iter())))
                .map(|(i, (d, (k, s)))| (i - d * (k - 1) - 1) / s + 1)
                .collect();
            let e_: Vec<usize> = k_
                .dims
                .iter()
                .zip(i_.iter().zip(d_.dims.iter()))
                .map(|(k, (i, d))| k * (i + d) / i)
                .collect();
            let mut tmp = prefix.clone();
            tmp.extend(i_.iter().map(|i| vec![1, *i]).collect::<Vec<Vec<usize>>>().iter().flatten());
            let mut tmp2 = prefix.clone();
            tmp2.extend(e_.iter().zip(i_.iter()).map(|(e, i)| vec![*e, *i]).collect::<Vec<Vec<usize>>>().iter().flatten());
            let mut tmp3 = prefix.clone();
            tmp3.extend(e_.iter().zip(i_.iter()).map(|(e, i)| *e * *i));
            let mut xup = self.reshape(tmp).expand(tmp2).reshape(tmp3);

            // Stride by dilation
            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(i_.iter().zip(d_.dims.iter())).map(|(k, (i, d))|{
                (0, k*(i+d))
            }));
            xup = xup.slice(tmp.iter().map(|e| (e.0 as isize, e.1 as isize)).collect::<Vec<(isize, isize)>>(), B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(i_.iter().zip(d_.dims.iter())).map(|(k, (i, d))| vec![*k, *i + *d]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter().zip(s_.dims.iter())).map(|(k, (o, s))| vec![(0, *k),(0, *o * *s)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = xup.slice(tmp.iter().map(|e| (e.0 as isize, e.1 as isize)).collect::<Vec<(isize, isize)>>(), B::Dtype::zero());

            // handle stride, and permute to move reduce to the end
            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter().zip(s_.dims.iter())).map(|(k, (o, s))| vec![*k, *o, *s]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter()).map(|(k, o)| vec![(0, *k), (0, *o), (0, 1)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = xup.slice(tmp.iter().map(|e| (e.0 as isize, e.1 as isize)).collect::<Vec<(isize, isize)>>(), B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(k_.dims.iter().zip(o_.iter()).map(|(k, o)| vec![*k, *o]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp: Vec<usize> = (0..prefix.len()).into_iter().collect();
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2 + 1));
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2));
            xup.permute(tmp)
        } else {
            let o_:Vec<usize> = i_.iter().zip(s_.dims.iter().zip(k_.dims.iter())).map(|(i, (s, k))|
                (*i+(*s-*k))*s
            ).collect();

            let mut tmp = slc_prefix.clone();
            tmp.extend(o_.iter().zip(s_.dims.iter()).map(|(o, s)| vec![(0, *o * *s)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            let mut xup = self.slice(tmp.iter().map(|e| (e.0 as isize, e.1 as isize)).collect::<Vec<(isize, isize)>>(), B::Dtype::zero());

            let mut tmp = prefix.clone();
            tmp.extend(o_.iter().zip(s_.dims.iter()).map(|(o, s)| vec![*o, *s]).collect::<Vec<Vec<usize>>>().iter().flatten());
            xup = xup.reshape(tmp);

            let mut tmp = slc_prefix.clone();
            tmp.extend(o_.iter().zip(k_.dims.iter()).map(|(o, k)| vec![(0, *o), (0, *k)]).collect::<Vec<Vec<(usize, usize)>>>().iter().flatten());
            xup = xup.slice(tmp.iter().map(|e| (e.0 as isize, e.1 as isize)).collect::<Vec<(isize, isize)>>(), B::Dtype::zero());

            let mut tmp: Vec<usize> = (0..prefix.len()).into_iter().collect();
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2));
            tmp.extend((0..k_.dims.len()).map(|i| prefix.len() + i * 2 + 1));
            xup.permute(tmp)
        };
        xup
    }

    #[rustfmt::skip]
    pub fn slice<A: Into<Vec<(isize, isize)>>>(&self, arg: A, const_value: B::Dtype) -> Self {
        let arg = arg.into();
        let self_shape = self.shape();
        let padding: Vec<(usize, usize)> = arg.iter().enumerate().map(|(i, p)|
            (0.max(p.0) as usize, 0.max(p.1 - self_shape[i] as isize) as usize)
        ).collect();
        let shrink: Vec<(usize, usize)> = arg.iter().enumerate().map(|(i, p)|
            (p.0 as usize + padding[i].0, p.1 as usize + padding[i].0)
        ).collect();
        self.pad(padding, const_value).shrink(shrink)
    }

    pub fn conv2d(
        &self,
        weigth: &Self,
        bias: Option<&Self>,
        groups: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
    ) -> Self {
        let [bs, cin_] = self.shape().dims[..2] else {
            panic!()
        };
        let [cout, cin] = weigth.shape().dims[..2] else {
            panic!()
        };
        let HW = weigth.shape().dims[2..].to_vec();
        let mut padding_ = vec![padding; 2 * HW.len()];
        let mut x =
            self.pad2d(padding_, B::Dtype::zero())
                ._pool(Shape::from(HW.clone()), stride, dilation);
        let rcout = cout / groups;
        let oyx = x.shape().dims[2..x.shape().len() - HW.len()].to_vec();
        //reshape(bs, groups, cin, 1, *oyx, *HW)
        let mut rsh_tmp = vec![bs, groups, cin, 1];
        rsh_tmp.extend(oyx.iter());
        rsh_tmp.extend(HW.iter());
        //expand(bs, groups, cin, rcout, *oyx, *HW)
        let mut exp_tmp = vec![bs, groups, cin, rcout];
        exp_tmp.extend(oyx.iter());
        exp_tmp.extend(HW.iter());
        //permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
        let mut permute_tmp = vec![0, 1, 3];
        permute_tmp.extend((0..oyx.len()).into_iter().map(|i| 4 + i));
        permute_tmp.push(2);
        permute_tmp.extend((0..HW.len()).into_iter().map(|i| 4 + oyx.len() + i));
        x = x.reshape(rsh_tmp).expand(exp_tmp).permute(permute_tmp);
        // ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)
        let mut w_rsh_tmp = vec![1, groups, rcout];
        w_rsh_tmp.extend(vec![1; oyx.len()]);
        w_rsh_tmp.push(cin);
        w_rsh_tmp.extend(HW.iter());
        let mut ret = x * weigth.reshape(w_rsh_tmp);
        for i in 0..oyx.len() + 1 {
            let reduce_i = -1 - (i as isize);
            ret = ret.sum_keepdim(reduce_i);
        }
        let mut ret_rsh_tmp = vec![bs, cout];
        ret_rsh_tmp.extend(oyx.iter());
        ret = ret.reshape(ret_rsh_tmp);
        if bias.is_none() {
            return ret;
        }
        // bias.reshape(1, -1, *[1] * len(HW))
        let bias = bias.unwrap();
        let mut b_rsh_tmp = vec![1, bias.shape().len()];
        b_rsh_tmp.extend(vec![1; HW.len()]);
        ret + bias.reshape(b_rsh_tmp)
    }

    pub fn t(&self) -> Self {
        self.transpose(1, 0)
    }
}

//TODO: Tests should be in a macro so that each backend can generate test.

#[test]
fn sum_axis() {
    use crate::prelude::*;
    let n = 2 * 3;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..n + 1)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [2, 3],
    );
    let y = t.sum(1);
    assert!(vec![6.0f32, 15.0f32] == y.to_vec());

    let n = 4 * 2 * 3 * 3;
    let mut t = Tensor::<Cpu>::from_vec(
        (1..n + 1)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [4, 2, 3, 3],
    );

    let y = t.sum(0);
    assert!(
        vec![
            112., 116., 120., 124., 128., 132., 136., 140., 144., 148., 152., 156., 160., 164.,
            168., 172., 176., 180.
        ] == y.to_vec()
    );

    let y = t.sum(1);
    assert!(
        vec![
            11., 13., 15., 17., 19., 21., 23., 25., 27., 47., 49., 51., 53., 55., 57., 59., 61.,
            63., 83., 85., 87., 89., 91., 93., 95., 97., 99., 119., 121., 123., 125., 127., 129.,
            131., 133., 135.
        ] == y.to_vec()
    );

    let y = t.sum(2);
    assert!(
        vec![
            12., 15., 18., 39., 42., 45., 66., 69., 72., 93., 96., 99., 120., 123., 126., 147.,
            150., 153., 174., 177., 180., 201., 204., 207.
        ] == y.to_vec()
    );

    let y = t.sum(3);
    assert!(
        vec![
            6., 15., 24., 33., 42., 51., 60., 69., 78., 87., 96., 105., 114., 123., 132., 141.,
            150., 159., 168., 177., 186., 195., 204., 213.
        ] == y.to_vec()
    );
}

#[test]
fn matmul() {
    let a = Tensor::<Cpu>::from_vec([1., 2., 3., 4., 5., 6.], [2, 3]);
    let b = Tensor::<Cpu>::from_vec([10., 11., 20., 21., 30., 31.], [3, 2]);
    &a + 1.0;
    let y = a.matmul(&b);
    assert!(vec![140., 146., 320., 335.] == y.to_vec());

    let a = Tensor::<Cpu>::from_vec(
        (1..=4 * 9)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [4, 9],
    );
    let b = Tensor::<Cpu>::from_vec(
        (1..=9 * 2)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [9, 2],
    );
    let y = a.matmul(&b);
    assert!(vec![525., 570., 1254., 1380., 1983., 2190., 2712., 3000.] == y.to_vec())
}

#[test]
fn pool() {
    let n = 9;
    let mut a = Tensor::<Cpu>::from_vec(
        (1..=n * n)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [n, n],
    );
    let k = Shape::from([3, 3]);
    let y = a._pool(k, 1, 1);
    assert!(
        vec![
            1., 2., 3., 10., 11., 12., 19., 20., 21., 2., 3., 4., 11., 12., 13., 20., 21., 22., 3.,
            4., 5., 12., 13., 14., 21., 22., 23., 4., 5., 6., 13., 14., 15., 22., 23., 24., 5., 6.,
            7., 14., 15., 16., 23., 24., 25., 6., 7., 8., 15., 16., 17., 24., 25., 26., 7., 8., 9.,
            16., 17., 18., 25., 26., 27., 10., 11., 12., 19., 20., 21., 28., 29., 30., 11., 12.,
            13., 20., 21., 22., 29., 30., 31., 12., 13., 14., 21., 22., 23., 30., 31., 32., 13.,
            14., 15., 22., 23., 24., 31., 32., 33., 14., 15., 16., 23., 24., 25., 32., 33., 34.,
            15., 16., 17., 24., 25., 26., 33., 34., 35., 16., 17., 18., 25., 26., 27., 34., 35.,
            36., 19., 20., 21., 28., 29., 30., 37., 38., 39., 20., 21., 22., 29., 30., 31., 38.,
            39., 40., 21., 22., 23., 30., 31., 32., 39., 40., 41., 22., 23., 24., 31., 32., 33.,
            40., 41., 42., 23., 24., 25., 32., 33., 34., 41., 42., 43., 24., 25., 26., 33., 34.,
            35., 42., 43., 44., 25., 26., 27., 34., 35., 36., 43., 44., 45., 28., 29., 30., 37.,
            38., 39., 46., 47., 48., 29., 30., 31., 38., 39., 40., 47., 48., 49., 30., 31., 32.,
            39., 40., 41., 48., 49., 50., 31., 32., 33., 40., 41., 42., 49., 50., 51., 32., 33.,
            34., 41., 42., 43., 50., 51., 52., 33., 34., 35., 42., 43., 44., 51., 52., 53., 34.,
            35., 36., 43., 44., 45., 52., 53., 54., 37., 38., 39., 46., 47., 48., 55., 56., 57.,
            38., 39., 40., 47., 48., 49., 56., 57., 58., 39., 40., 41., 48., 49., 50., 57., 58.,
            59., 40., 41., 42., 49., 50., 51., 58., 59., 60., 41., 42., 43., 50., 51., 52., 59.,
            60., 61., 42., 43., 44., 51., 52., 53., 60., 61., 62., 43., 44., 45., 52., 53., 54.,
            61., 62., 63., 46., 47., 48., 55., 56., 57., 64., 65., 66., 47., 48., 49., 56., 57.,
            58., 65., 66., 67., 48., 49., 50., 57., 58., 59., 66., 67., 68., 49., 50., 51., 58.,
            59., 60., 67., 68., 69., 50., 51., 52., 59., 60., 61., 68., 69., 70., 51., 52., 53.,
            60., 61., 62., 69., 70., 71., 52., 53., 54., 61., 62., 63., 70., 71., 72., 55., 56.,
            57., 64., 65., 66., 73., 74., 75., 56., 57., 58., 65., 66., 67., 74., 75., 76., 57.,
            58., 59., 66., 67., 68., 75., 76., 77., 58., 59., 60., 67., 68., 69., 76., 77., 78.,
            59., 60., 61., 68., 69., 70., 77., 78., 79., 60., 61., 62., 69., 70., 71., 78., 79.,
            80., 61., 62., 63., 70., 71., 72., 79., 80., 81.
        ] == y.to_vec(), "{y:?}"
    );
}

#[test]
fn conv2d() {
    let mut a = Tensor::<Cpu>::from_vec(
        (1..=9 * 9)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [1, 1, 9, 9],
    );
    let mut b = Tensor::<Cpu>::from_vec(
        (1..=3 * 3)
            .map(|e| f32::from_usize(e).unwrap())
            .collect::<Vec<f32>>(),
        [1, 1, 3, 3],
    );
    let r = a.conv2d(&b, None, 1, 1, 1, 0);
    assert!(
        vec![
            663., 708., 753., 798., 843., 888., 933., 1068., 1113., 1158., 1203., 1248., 1293.,
            1338., 1473., 1518., 1563., 1608., 1653., 1698., 1743., 1878., 1923., 1968., 2013.,
            2058., 2103., 2148., 2283., 2328., 2373., 2418., 2463., 2508., 2553., 2688., 2733.,
            2778., 2823., 2868., 2913., 2958., 3093., 3138., 3183., 3228., 3273., 3318., 3363.
        ] == r.to_vec()
    )
}
