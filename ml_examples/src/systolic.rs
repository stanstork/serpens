/*
Predicting Systolic Blood Pressure from Age and Weight
The data (x1, x2, x3) are for each patient.

x1 = Systolic Blood Pressure
x2 = Age in years
x3 = Weight in pounds
*/

use std::io::Error;

use cayley::{
    ml::{common::Common, linear_regression::LinearRegression, regression::Regression},
    reader::Reader,
};
use linear::vector::{shape::Shape, vector::Vector};

pub fn predicting_systolic_blood_pressure() {
    let data: Result<Vec<Vec<f64>>, Error> =
        Reader::read_nd_csv("/home/stan/serpens/ml_examples/test_data/mlr02.csv", 3);
    match data {
        Ok(data) => {
            let pressure: Vec<f64> = data[0].clone();
            let age: Vec<f64> = data[1].clone();
            let weight: Vec<f64> = data[2].clone();

            let mut pressure_weight: LinearRegression<f64> = LinearRegression::new(
                Vector::new(pressure, Shape::Col),
                Vector::new(weight.clone(), Shape::Col),
            );

            let mut age_weight: LinearRegression<f64> = LinearRegression::new(
                Vector::new(age, Shape::Col),
                Vector::new(weight.clone(), Shape::Col),
            );

            pressure_weight.train();
            age_weight.train();

            Common::plot_2d(
                pressure_weight.x(),
                pressure_weight.y(),
                pressure_weight.regression().unwrap(),
                "/home/stan/serpens/ml_examples/images/systolic_pressure_weight.png",
            );
            Common::plot_2d(
                age_weight.x(),
                age_weight.y(),
                age_weight.regression().unwrap(),
                "/home/stan/serpens/ml_examples/images/systolic_age_weight.png",
            );

            println!(
                "r2 for pressure weight: {}",
                pressure_weight.r_squared().unwrap()
            );
            println!("r2 for age weight: {}", age_weight.r_squared().unwrap());
        }
        Err(e) => panic!("ERROR: {}", e),
    }
}
