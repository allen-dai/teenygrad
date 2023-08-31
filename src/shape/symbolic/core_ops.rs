use super::{num, sum, DivNode, MulNode, Node};

///////////////////////////////////////////// for self is borrowed
impl core::ops::Add for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn add(self, rhs: Self) -> Self::Output {
        self._add(rhs._clone())
    }
}

impl core::ops::Sub for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn sub(self, rhs: Self) -> Self::Output {
        self._sub(rhs._clone())
    }
}

impl core::ops::Mul for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn mul(self, rhs: Self) -> Self::Output {
        self._mul(rhs._clone())
    }
}

impl core::ops::Div for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn div(self, rhs: Self) -> Self::Output {
        self._div(rhs._clone(), None)
    }
}

impl core::ops::Add<isize> for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn add(self, rhs: isize) -> Self::Output {
        self._add(num(rhs))
    }
}

impl core::ops::Sub<isize> for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn sub(self, rhs: isize) -> Self::Output {
        self._sub(num(rhs))
    }
}

impl core::ops::Mul<isize> for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn mul(self, rhs: isize) -> Self::Output {
        self._mul(num(rhs))
    }
}

impl core::ops::Div<isize> for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn div(self, rhs: isize) -> Self::Output {
        self._div(num(rhs), None)
    }
}

impl core::ops::Neg for &Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn neg(self) -> Self::Output {
        self * &num(-1)
    }
}
//////////////////////////////////////////////////////////

///////////////////////////////////////////// for self is moved
impl core::ops::Add for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn add(self, rhs: Self) -> Self::Output {
        self._add(rhs)
    }
}

impl core::ops::Sub for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn sub(self, rhs: Self) -> Self::Output {
        self._sub(rhs)
    }
}

impl core::ops::Mul for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn mul(self, rhs: Self) -> Self::Output {
        self._mul(rhs)
    }
}

impl core::ops::Div for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn div(self, rhs: Self) -> Self::Output {
        self._div(rhs, None)
    }
}

impl core::ops::Add<isize> for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn add(self, rhs: isize) -> Self::Output {
        self._add(num(rhs))
    }
}

impl core::ops::Sub<isize> for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn sub(self, rhs: isize) -> Self::Output {
        self._sub(num(rhs))
    }
}

impl core::ops::Mul<isize> for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn mul(self, rhs: isize) -> Self::Output {
        self._mul(num(rhs))
    }
}

impl core::ops::Div<isize> for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn div(self, rhs: isize) -> Self::Output {
        self._div(num(rhs), None)
    }
}

impl core::ops::Neg for Box<dyn Node> {
    type Output = Box<dyn Node>;

    fn neg(self) -> Self::Output {
        self * num(-1)
    }
}
//////////////////////////////////////////////////////////

impl core::fmt::Display for &dyn Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
    }
}

impl core::fmt::Display for Box<dyn Node> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
        // write!(f, "<{}>", self.key())
    }
}
