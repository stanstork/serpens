use std::ops::{Add, Div, DivAssign, Mul, Sub, SubAssign};

use linear::{
    matrix::matrix::Matrix,
    num::Num,
    vector::{shape::Shape, vector::Vector},
};

use crate::alg::gaussian_elimination::GaussianElimination;

use super::regression::{self, Regression};

pub struct MultLinearRegression<T> {
    x: Matrix<T>,
    y: Vector<T>,
    regression: Option<Vector<T>>,
    r_squared: Option<T>,
    weights: Option<Vector<T>>,
}

impl<'a, T> Regression<T> for MultLinearRegression<T>
where
    T: Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + Sub<Output = T>
        + SubAssign
        + DivAssign
        + Num,
{
    fn train(&mut self) {
        let xt_t_prod: Result<Matrix<T>, &str> = self.x().transpose().product(self.x());
        match xt_t_prod {
            Ok(xt_x_prod) => {
                let xt_y_prod = self.x().transpose().vector_product(self.y());
                match xt_y_prod {
                    Ok(xt_y_prod) => {
                        let weights: Option<Vector<T>> =
                            GaussianElimination::solve(xt_x_prod, xt_y_prod);
                        match weights {
                            Some(weights) => {
                                let regression: Result<Vector<T>, &str> =
                                    self.x().vector_product(&weights);
                                match regression {
                                    Ok(regression) => {
                                        self.set_regression(regression);
                                        self.set_weights(weights);
                                    }
                                    Err(e) => println!("ERROR: {}", e),
                                }
                            }
                            None => println!("ERROR"),
                        }
                    }
                    Err(e) => println!("ERROR: {}", e),
                }
            }
            Err(e) => println!("ERROR: {}", e),
        }
    }

    fn r_squared(&mut self) -> Option<T> {
        todo!()
    }

    fn plot(&self, output: &str) {
        todo!()
    }
}

impl<T> MultLinearRegression<T>
where
    T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Num,
{
    pub fn new(data: Vec<Vec<T>>) -> Self {
        let x: Matrix<T> = Matrix::new(
            &data
                .iter()
                .take(data.len() - 1)
                .map(|c| c.clone())
                .collect(),
        );
        let y: Vector<T> = Vector::new(data.last().unwrap().clone(), Shape::Col);

        MultLinearRegression {
            x,
            y,
            regression: None,
            r_squared: None,
            weights: None,
        }
    }

    pub fn y(&self) -> &Vector<T> {
        &self.y
    }

    pub fn x(&self) -> &Matrix<T> {
        &self.x
    }

    pub fn set_regression(&mut self, regression: Vector<T>) {
        self.regression = Some(regression);
    }

    pub fn set_weights(&mut self, weights: Vector<T>) {
        self.weights = Some(weights);
    }
}

#[cfg(test)]
mod test {
    use std::io::Error;

    use crate::reader::Reader;

    use super::MultLinearRegression;

    #[test]
    fn test_init() {
        let data: Result<Vec<Vec<f64>>, Error> = Reader::read_nd_csv("test_data/data_2d.csv", 3);
        match data {
            Ok(data) => {
                let mlr: MultLinearRegression<f64> = MultLinearRegression::new(data);

                println!("{:?}", mlr.x());
                println!("{:?}", mlr.y());
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
