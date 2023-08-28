#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct TensorId(pub(crate) usize);

pub(crate) fn tensor_id() -> TensorId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    TensorId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
