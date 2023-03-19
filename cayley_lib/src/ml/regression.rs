pub trait Regression<T> {
    fn train(&mut self);
    fn r_squared(&mut self) -> Option<T>;
    fn plot(&self, output: &str);
}
