use teenygrad::prelude::*;
use teenygrad::shape::shapetracker::{view::*, ShapeTracker};

#[test]
pub fn test_reshape_doesnt_multiview() {
    let mut st = ShapeTracker::new(
        &[256, 256, 2, 2, 2, 2, 2, 256, 8, 2],
        Some(vec![view!(
            [256, 256, 2, 2, 2, 2, 2, 256, 8, 2],
            [0, 8, 0, 4, 0, 0, 2, 16384, 2048, 1],
            0
        )]),
    );
    st.reshape(&[128, 2, 256, 2, 2, 2, 2, 2, 256, 8, 2]);
    assert!(st.views.len() == 1)
}

#[test]
pub fn test_real_doesnt_simplify_1() {
    let mut st = ShapeTracker::new(
        &[8, 6, 11],
        Some(vec![
            view!([8, 3, 1, 2, 11, 1], [33, 11, 0, 0, 1, 0]),
            view!([8, 6, 11], [66, 11, 1]),
        ]),
    );

    assert!(
        st.real_strides(false) == vec![Some(33), None, Some(1)],
        // "{:?} != {:?}",
        // st.real_strides(false),
        // vec![Some(33), None, Some(1)]
    );
}

#[test]
pub fn test_real_doesnt_simplify_2() {
    let mut st = ShapeTracker::new(
        &[4, 4, 3, 3],
        Some(vec![
            view!([2, 2, 4, 3, 3], [72, 9, 18, -3, -1], 8),
            view!([4, 4, 3, 3], [36, 9, 3, 1]),
        ]),
    );

    assert!(st.real_strides(false) == vec![None, Some(18), Some(-3), Some(-1)],);
}


#[test]
pub fn test_realstrides() {
    let mut st = ShapeTracker::new(
        &[16, 32, 4],
        Some(vec![
            view!([2048], [1], 0, [(0, 512)]),
            view!([16,32,4], [128,4,1]),
        ]),
    );
    let rs = st.real_strides(false);
    assert!(rs == vec![None, Some(4), Some(1)]);
}
