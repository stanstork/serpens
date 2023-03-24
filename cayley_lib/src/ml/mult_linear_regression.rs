use std::ops::{Add, Div, DivAssign, Mul, Sub, SubAssign};

use linear::{
    matrix::matrix::Matrix,
    num::Num,
    vector::{shape::Shape, vector::Vector},
};

use crate::alg::gaussian_elimination::GaussianElimination;

use super::{common::Common, regression::Regression};

pub struct MultLinearRegression<T> {
    x: Matrix<T>,
    y: Vector<T>,
    regression: Option<Vector<T>>,
    r_squared: Option<T>,
    weights: Option<Vector<T>>,
}

impl<T> Regression<T> for MultLinearRegression<T>
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
        if self.r_squared.is_none() && self.regression.is_some() {
            self.r_squared = Common::r_squared(self.y(), self.regression.as_ref().unwrap());
            return self.r_squared;
        }
        self.r_squared
    }
}

impl<T> MultLinearRegression<T>
where
    T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Num,
{
    pub fn new(data: Vec<Vec<T>>) -> Self {
        let mut elements: Vec<Vec<T>> = vec![vec![]; data[0].len()];
        for (i, item) in elements.iter_mut().enumerate().take(data[0].len()) {
            for el in data.iter().take(data.len() - 1) {
                item.push(el[i]);
            }
        }

        let x: Matrix<T> = Matrix::new(&elements);
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

    pub fn weights(&self) -> Option<&Vector<T>> {
        self.weights.as_ref()
    }

    pub fn regression(&self) -> Option<&Vector<T>> {
        self.regression.as_ref()
    }
}

#[cfg(test)]
mod test {
    use std::io::Error;

    use linear::vector::{shape::Shape, vector::Vector};

    use crate::{
        ml::{common::Common, regression::Regression},
        reader::Reader,
    };

    use super::MultLinearRegression;

    use rand::{rngs::ThreadRng, Rng};

    #[test]
    fn test_train() {
        let data: Result<Vec<Vec<f64>>, Error> =
            Reader::read_nd_csv("/home/stan/serpens/cayley_lib/test_data/data_2d.csv", 3);
        match data {
            Ok(data) => {
                let mut mlr: MultLinearRegression<f64> = MultLinearRegression::new(data);

                println!("{:?}", mlr.x());
                println!("{:?}", mlr.y());

                mlr.train();

                println!("{:?}", mlr.weights().unwrap());
                println!("{:?}", mlr.r_squared());

                assert!(mlr.r_squared().is_some());
                assert!(mlr.r_squared().unwrap() > 0.99);
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    #[test]
    fn test_noise() {
        let data: Result<Vec<Vec<f64>>, Error> =
            Reader::read_nd_csv("/home/stan/serpens/cayley_lib/test_data/data_2d.csv", 3);
        match data {
            Ok(data) => {
                let mut mlr_data: Vec<Vec<f64>> = vec![vec![]; data.len() + 1];
                for i in 0..(data.len() - 1) {
                    mlr_data[i] = data[i].clone();
                }

                let mut rng: ThreadRng = rand::thread_rng();
                mlr_data[data.len() - 1] = (0..data[0].len())
                    .map(|_| rng.gen_range(0.0..150.0))
                    .collect();

                mlr_data[data.len()] = data.last().unwrap().clone();

                let mut mlr: MultLinearRegression<f64> = MultLinearRegression::new(mlr_data);

                println!("{:?}", mlr.x());
                println!("{:?}", mlr.y());

                mlr.train();

                println!("{:?}", mlr.weights().unwrap());
                println!("{:?}", mlr.r_squared());

                assert!(mlr.r_squared().is_some());
                assert!(mlr.r_squared().unwrap() > 0.99);
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    #[test]
    fn test_poly() {
        let data: Result<Vec<Vec<f64>>, Error> =
            Reader::read_nd_csv("/home/stan/serpens/cayley_lib/test_data/data_poly.csv", 2);
        match data {
            Ok(data) => {
                let ones: Vec<f64> = vec![1.0; data[0].len()];
                let x: Vec<f64> = data[0].clone();
                let x2: Vec<f64> = x.iter().map(|e| e * e).collect();

                let mut mlr: MultLinearRegression<f64> =
                    MultLinearRegression::new(vec![ones, x, x2, data[1].clone()]);

                mlr.train();

                assert!(mlr.regression().is_some());

                Common::plot_2d(
                    &Vector::new(mlr.x().get_col(1), Shape::Col).sorted(),
                    &mlr.y().sorted(),
                    &mlr.regression().unwrap().sorted(),
                    "images/poly.png",
                );
                // validate image manually

                assert!(mlr.r_squared().unwrap() > 0.999);

                println!("{:?}", mlr.r_squared().unwrap());
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
