#[derive(Debug)]
pub struct MultivariatePPData {
    pub timestamps: Vec<Vec<f64>>, // Get the ownership of the timestamps data because timestamp data should be treated only from this struct
    pub obs_window: (f64, f64),
}
