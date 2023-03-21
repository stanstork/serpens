use std::ops::{Add, Div, Mul, Sub};

use linear::{num::Num, vector::vector::Vector};
use plotters::{
    prelude::{BitMapBackend, ChartBuilder, Circle, IntoDrawingArea, LabelAreaPosition},
    series::LineSeries,
    style::{BLUE, RED, WHITE},
};

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

    pub fn plot_2d<T>(x: &Vector<T>, y: &Vector<T>, r: &Vector<T>, output: &str)
    where
        T: Mul<Output = T> + Add<Output = T> + Num,
    {
        let size: (u32, u32) = (1200, 800);
        let root_area = BitMapBackend::new(output, size).into_drawing_area();

        root_area.fill(&WHITE).unwrap();

        let min_x = x.min().as_f64();
        let min_y = y.min().as_f64();
        let max_x = x.max().as_f64();
        let max_y = y.max().as_f64();

        let label_area_size: i32 = 40;
        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, label_area_size)
            .set_label_area_size(LabelAreaPosition::Bottom, label_area_size)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();

        let circle_size = 5;

        ctx.draw_series(
            x.elements()
                .iter()
                .zip(y.elements().iter())
                .map(|e| (e.0.as_f64(), e.1.as_f64()))
                .collect::<Vec<(f64, f64)>>()
                .iter()
                .map(|point| Circle::new(*point, circle_size, BLUE)),
        )
        .unwrap();
        ctx.draw_series(LineSeries::new(
            x.elements()
                .iter()
                .zip(r.elements().iter())
                .map(|e| (e.0.as_f64(), e.1.as_f64()))
                .collect::<Vec<(f64, f64)>>(),
            &RED,
        ))
        .unwrap();
    }
}
