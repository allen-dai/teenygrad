use crate::dtype;

pub struct LocalBuffer {
    pub name: String,
    pub size: usize,
    pub dtype: dtype::DType,
    pub realized: bool,
}

impl core::fmt::Display for LocalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "localbuffer<{}[{}]>", self.name, self.size)
    }
}

pub struct LinearizerOptions {
    support_float4: bool,
    support_float4_alu: bool,
    has_local: bool,
    global_max: Option<Vec<isize>>,
    local_max: Option<Vec<isize>>,
}
