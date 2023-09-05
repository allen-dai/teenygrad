use super::{num, sum, DivNode, MulNode, Node, ArcNode};
use std::sync::Arc;

///////////////////////////////////////////// for self is borrowed
impl core::ops::Add for &ArcNode {
    type Output = ArcNode;

    fn add(self, rhs: Self) -> Self::Output {
        self._add(rhs.clone())
    }
}

impl core::ops::Sub for &ArcNode {
    type Output = ArcNode;

    fn sub(self, rhs: Self) -> Self::Output {
        self._sub(rhs.clone())
    }
}

impl core::ops::Mul for &ArcNode {
    type Output = ArcNode;

    fn mul(self, rhs: Self) -> Self::Output {
        self._mul(rhs.clone())
    }
}

impl core::ops::Div for &ArcNode {
    type Output = ArcNode;

    fn div(self, rhs: Self) -> Self::Output {
        self._div(rhs.clone(), None)
    }
}

impl core::ops::Rem for &ArcNode {
    type Output = ArcNode;

    fn rem(self, rhs: Self) -> Self::Output {
        self._mod(rhs.clone())
    }
}

impl core::ops::Add<isize> for &ArcNode {
    type Output = ArcNode;

    fn add(self, rhs: isize) -> Self::Output {
        self._add(num(rhs))
    }
}

impl core::ops::Sub<isize> for &ArcNode {
    type Output = ArcNode;

    fn sub(self, rhs: isize) -> Self::Output {
        self._sub(num(rhs))
    }
}

impl core::ops::Mul<isize> for &ArcNode {
    type Output = ArcNode;

    fn mul(self, rhs: isize) -> Self::Output {
        self._mul(num(rhs))
    }
}

impl core::ops::Div<isize> for &ArcNode {
    type Output = ArcNode;

    fn div(self, rhs: isize) -> Self::Output {
        self._div(num(rhs), None)
    }
}

impl core::ops::Rem<isize> for &ArcNode {
    type Output = ArcNode;

    fn rem(self, rhs: isize) -> Self::Output {
        self._mod(num(rhs))
    }
}

impl core::ops::Neg for &ArcNode {
    type Output = ArcNode;

    fn neg(self) -> Self::Output {
        self * -1
    }
}
//////////////////////////////////////////////////////////

///////////////////////////////////////////// for self is moved
impl core::ops::Add for ArcNode {
    type Output = ArcNode;

    fn add(self, rhs: Self) -> Self::Output {
        self._add(rhs)
    }
}

impl core::ops::Sub for ArcNode {
    type Output = ArcNode;

    fn sub(self, rhs: Self) -> Self::Output {
        self._sub(rhs)
    }
}

impl core::ops::Mul for ArcNode {
    type Output = ArcNode;

    fn mul(self, rhs: Self) -> Self::Output {
        self._mul(rhs)
    }
}

impl core::ops::Div for ArcNode {
    type Output = ArcNode;

    fn div(self, rhs: Self) -> Self::Output {
        self._div(rhs, None)
    }
}

impl core::ops::Rem for ArcNode {
    type Output = ArcNode;

    fn rem(self, rhs: Self) -> Self::Output {
        self._mod(rhs)
    }
}

impl core::ops::Add<isize> for ArcNode {
    type Output = ArcNode;

    fn add(self, rhs: isize) -> Self::Output {
        self._add(num(rhs))
    }
}

impl core::ops::Sub<isize> for ArcNode {
    type Output = ArcNode;

    fn sub(self, rhs: isize) -> Self::Output {
        self._sub(num(rhs))
    }
}

impl core::ops::Mul<isize> for ArcNode {
    type Output = ArcNode;

    fn mul(self, rhs: isize) -> Self::Output {
        self._mul(num(rhs))
    }
}

impl core::ops::Div<isize> for ArcNode {
    type Output = ArcNode;

    fn div(self, rhs: isize) -> Self::Output {
        self._div(num(rhs), None)
    }
}

impl core::ops::Rem<isize> for ArcNode {
    type Output = ArcNode;

    fn rem(self, rhs: isize) -> Self::Output {
        self._mod(num(rhs))
    }
}

impl core::ops::Neg for ArcNode {
    type Output = ArcNode;

    fn neg(self) -> Self::Output {
        self * num(-1)
    }
}
//////////////////////////////////////////////////////////

impl core::fmt::Display for ArcNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
    }
}

impl PartialEq for dyn Node {
    fn eq(&self, other: &Self) -> bool {
        if self.is_num() && other.is_num() {
            return self.num_val().unwrap() == other.num_val().unwrap();
        }
        self.key() == other.key()
    }
}

impl Eq for dyn Node {}

impl std::hash::Hash for dyn Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

