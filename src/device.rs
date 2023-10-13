use crate::runtime::RawBuffer;

lazy_static::lazy_static! {
    pub static ref DEVICE: &'static str = "GPU";
}


// #[derive(Debug, Clone)]
// pub struct Device {
//     name: String,
// }
//
// unsafe impl Send for Device {}
// unsafe impl Sync for Device {}
//
// impl Device {
// }
//

pub trait Device {
    fn buffer() -> RawBuffer {
        unimplemented!()
    }

    fn render() -> String {
        unimplemented!()
    }

    fn exec() {
        unimplemented!()
    }
}
