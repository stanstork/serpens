use std::ops::{Add, Div, Mul, Sub};

use linear::{num::Num, vector::vector::Vector};

pub struct Common {}

impl Common {
    pub fn r_squared<T>(y: &Vector<T>, regression: &Vector<T>) -> Option<T>
    where
        T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Num,
    {
        let rss: Result<Vector<T>, &str> = y.add_vector(&regression.mul(T::minus_one()));
        match rss {
            Ok(rss) => {
                let tss: Vector<T> = y.add(y.mean().mul(T::minus_one()));
                match rss.dot(&rss) {
                    Ok(rss_dot) => match tss.dot(&tss) {
                        Ok(tss_dot) => {
                            let r_squared: T = T::one() - rss_dot / tss_dot;
                            return Some(r_squared);
                        }
                        Err(e) => println!("ERROR: {}", e),
                    },
                    Err(e) => println!("ERROR: {}", e),
                }
            }
            Err(e) => println!("ERROR: {}", e),
        }
        None
    }
}
