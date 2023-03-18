use std::ops::{Add, Div, Mul, Sub};

use linear::{num::Num, vector::vector::Vector};
use plotters::prelude::*;

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

    pub fn plot(&self, output: &str) {
        let size: (u32, u32) = (1200, 800);
        let root_area = BitMapBackend::new(output, size).into_drawing_area();

        root_area.fill(&WHITE).unwrap();

        let min_x = self.x().min().as_f64();
        let min_y = self.y().min().as_f64();
        let max_x = self.x().max().as_f64();
        let max_y = self.y().max().as_f64();

        let label_area_size: i32 = 40;
        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, label_area_size)
            .set_label_area_size(LabelAreaPosition::Bottom, label_area_size)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();

        let circle_size = 5;

        ctx.draw_series(
            self.x()
                .elements()
                .iter()
                .zip(self.y().elements().iter())
                .map(|e| (e.0.as_f64(), e.1.as_f64()))
                .collect::<Vec<(f64, f64)>>()
                .iter()
                .map(|point| Circle::new(*point, circle_size, &BLUE)),
        )
        .unwrap();
        ctx.draw_series(LineSeries::new(
            self.x()
                .elements()
                .iter()
                .zip(self.regression().unwrap().elements().iter())
                .map(|e| (e.0.as_f64(), e.1.as_f64()))
                .collect::<Vec<(f64, f64)>>(),
            &RED,
        ))
        .unwrap();
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
        let data: Result<(Vector<f64>, Vector<f64>), Error> =
            Reader::read_2d_csv("test_data/data_1d.csv");
        match data {
            Ok(data) => {
                let mut lr: LinearRegression<f64> = LinearRegression::new(data.0, data.1);

                lr.train();
                lr.plot("images/2.6.png");
                // validate image manually
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
