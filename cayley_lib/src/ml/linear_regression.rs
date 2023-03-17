use std::ops::{Add, Div, Mul, Sub};

use linear::{num::Num, vector::vector::Vector};

pub struct LinearRegression<T> {
    x: Vector<T>,
    y: Vector<T>,
    coefficient: Option<T>,
    constant: Option<T>,
    regression: Option<Vector<T>>,
    r_squared: Option<T>,
}

impl<T> LinearRegression<T>
where
    T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Num,
{
    pub fn new(x: Vector<T>, y: Vector<T>) -> Self {
        LinearRegression {
            x,
            y,
            coefficient: None,
            constant: None,
            regression: None,
            r_squared: None,
        }
    }

    pub fn train(&mut self) {
        match self.x().dot(self.x()) {
            Ok(xx_dot) => {
                let denominator: T = xx_dot - (self.x().mean() * self.x().sum());
                match self.x().dot(self.y()) {
                    Ok(xy_dot) => {
                        let coefficient: T =
                            (xy_dot - (self.y().mean() * self.x().sum())) / denominator;
                        let constant: T =
                            (self.y().mean() * xx_dot - (self.x().mean() * xy_dot)) / denominator;
                        let regression: Vector<T> = self.x().mul(coefficient).add(constant);

                        self.set_coefficient(coefficient);
                        self.set_constant(constant);
                        self.set_regression(regression);
                    }
                    Err(e) => println!("ERROR: {}", e),
                }
            }
            Err(e) => println!("ERROR: {}", e),
        }
    }

    pub fn r_squared(&mut self) -> Option<T> {
        if self.r_squared.is_none() && self.regression.is_some() {
            let rss: Result<Vector<T>, &str> = self
                .y()
                .add_vector(&self.regression.as_ref().unwrap().mul(T::minus_one()));
            match rss {
                Ok(rss) => {
                    let tss: Vector<T> = self
                        .y()
                        .add(self.regression.as_ref().unwrap().mean().mul(T::minus_one()));
                    match rss.dot(&rss) {
                        Ok(rss_dot) => match tss.dot(&tss) {
                            Ok(tss_dot) => {
                                let r_squared: T = T::one() - rss_dot / tss_dot;
                                self.r_squared = Some(r_squared);
                                return self.r_squared;
                            }
                            Err(e) => println!("ERROR: {}", e),
                        },
                        Err(e) => println!("ERROR: {}", e),
                    }
                }
                Err(e) => println!("ERROR: {}", e),
            }
        }
        return self.r_squared;
    }

    pub fn plot(&self) {
        todo!()
    }

    pub fn x(&self) -> &Vector<T> {
        &self.x
    }

    pub fn y(&self) -> &Vector<T> {
        &self.y
    }

    pub fn regression(&self) -> Option<&Vector<T>> {
        self.regression.as_ref()
    }

    fn set_coefficient(&mut self, coefficient: T) {
        self.coefficient = Some(coefficient);
    }

    fn set_constant(&mut self, constant: T) {
        self.constant = Some(constant);
    }

    fn set_regression(&mut self, regression: Vector<T>) {
        self.regression = Some(regression);
    }
}

#[cfg(test)]
mod test {
    use std::io::Error;

    use linear::vector::vector::Vector;

    use crate::reader::Reader;

    use super::LinearRegression;

    #[test]
    fn test_train() {
        let data: Result<(Vector<f64>, Vector<f64>), Error> =
            Reader::read_2d_csv("test_data/data_1d.csv");
        match data {
            Ok(data) => {
                let mut lr: LinearRegression<f64> = LinearRegression::new(data.0, data.1);

                println!("{:?}", lr.x());
                println!("{:?}", lr.y());

                lr.train();

                println!("{:?}", lr.r_squared());
                println!("{:?}", lr.regression());
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
