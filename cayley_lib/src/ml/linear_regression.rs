use std::ops::{Add, Div, Mul, Sub};

use linear::{num::Num, vector::vector::Vector};

use super::{common::Common, regression::Regression};

impl<T> Regression<T> for LinearRegression<T>
where
    T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Num,
{
    fn train(&mut self) {
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

    fn r_squared(&mut self) -> Option<T> {
        if self.r_squared.is_none() && self.regression.is_some() {
            self.r_squared = Common::r_squared(self.y(), self.regression.as_ref().unwrap());
            return self.r_squared;
        }
        self.r_squared
    }
}

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

    use linear::vector::{shape::Shape, vector::Vector};

    use crate::{
        ml::{common::Common, linear_regression::Regression},
        reader::Reader,
    };

    use super::LinearRegression;

    #[test]
    fn test_train() {
        let data: Result<Vec<Vec<f64>>, Error> = Reader::read_nd_csv("test_data/data_1d.csv", 2);
        match data {
            Ok(data) => {
                let mut lr: LinearRegression<f64> = LinearRegression::new(
                    Vector::new(data[0].clone(), Shape::Col),
                    Vector::new(data[1].clone(), Shape::Col),
                );

                assert_eq!(100, lr.x().elements().len());
                assert_eq!(100, lr.y().elements().len());

                assert_eq!(None, lr.coefficient);
                assert_eq!(None, lr.constant);

                lr.train();

                assert!(lr.coefficient.is_some());
                assert!(lr.constant.is_some());
                assert!(lr.r_squared().is_some());
                assert!(lr.r_squared().unwrap() > 0.99);
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    #[test]
    fn test_plot() {
        let data: Result<Vec<Vec<f64>>, Error> = Reader::read_nd_csv("test_data/data_1d.csv", 2);
        match data {
            Ok(data) => {
                let mut lr: LinearRegression<f64> = LinearRegression::new(
                    Vector::new(data[0].clone(), Shape::Col),
                    Vector::new(data[1].clone(), Shape::Col),
                );

                lr.train();
                Common::plot_2d(lr.x(), lr.y(), lr.regression().unwrap(), "images/2.6.png");
                // validate image manually
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
