use std::{
    fmt::Display,
    ops::{AddAssign, MulAssign},
};

use rand::distributions::uniform::SampleUniform;

pub trait Num: AddAssign + MulAssign + Copy + Sized + Display + PartialOrd + SampleUniform {
    fn zero() -> Self;
    fn one() -> Self;
    fn minus_one() -> Self;
    fn from_usize(u: usize) -> Self;
    fn absolute(&self) -> Self;
    fn from_str(s: &str) -> Self;
    fn get_min(&self, other: &Self) -> Self;
    fn get_max(&self, other: &Self) -> Self;
    fn max() -> Self;
    fn min() -> Self;
    fn as_f64(&self) -> f64;
}

impl Num for i32 {
    fn zero() -> Self {
        0
    }

    fn from_usize(u: usize) -> Self {
        u as i32
    }

    fn minus_one() -> Self {
        -1
    }

    fn one() -> Self {
        1
    }

    fn absolute(&self) -> Self {
        self.abs()
    }

    fn from_str(s: &str) -> Self {
        s.parse().unwrap()
    }

    fn get_min(&self, other: &i32) -> Self {
        *self.min(other)
    }

    fn max() -> Self {
        i32::MAX
    }

    fn min() -> Self {
        i32::MIN
    }

    fn get_max(&self, other: &Self) -> Self {
        *self.max(other)
    }

    fn as_f64(&self) -> f64 {
        *self as f64
    }
}

impl Num for f64 {
    fn zero() -> Self {
        0.0
    }

    fn from_usize(u: usize) -> Self {
        u as f64
    }

    fn minus_one() -> Self {
        -1.0
    }

    fn one() -> Self {
        1.0
    }

    fn absolute(&self) -> Self {
        self.abs()
    }

    fn from_str(s: &str) -> Self {
        s.parse().unwrap()
    }

    fn get_min(&self, other: &f64) -> Self {
        self.min(*other)
    }

    fn max() -> Self {
        f64::MAX
    }

    fn min() -> Self {
        f64::MIN
    }

    fn get_max(&self, other: &Self) -> Self {
        self.max(*other)
    }

    fn as_f64(&self) -> f64 {
        *self
    }
}
