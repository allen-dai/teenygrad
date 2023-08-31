#[macro_export]
macro_rules! approx_eq {
    ($Lhs:expr, $Rhs:expr) => {{
        let lhs = $Lhs.to_vec();
        let rhs = $Rhs.map(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
        lhs.iter()
            .zip(rhs.iter())
            .for_each(|(l, r)| assert!(float_cmp::approx_eq!(f32, *l, *r, ulps = 2), "{l} != {r}"));
    }};
    ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
        let lhs = $Lhs.to_vec();
        let rhs = $Rhs.map(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
        lhs.iter()
            .zip(rhs.iter())
            .for_each(|(l, r)| assert!(float_cmp::approx_eq!(f32, *l, *r, ulps = $Tolerance), "{l} != {r}"));
    }};
}
