use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LabelAreaPosition},
    series::LineSeries,
    style::{RED, WHITE},
};

pub struct Plotter {}

impl Plotter {
    pub fn plot_series(x: &Vec<f64>, y: &Vec<f64>, output: &str) {
        let size: (u32, u32) = (1200, 800);
        let root_area = BitMapBackend::new(output, size).into_drawing_area();

        root_area.fill(&WHITE).unwrap();

        let min_x = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let min_y = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_x = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let max_y = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let label_area_size: i32 = 40;
        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, label_area_size)
            .set_label_area_size(LabelAreaPosition::Bottom, label_area_size)
            .build_cartesian_2d((-1.0 + min_x)..(max_x + 1.0), (-1.0 + min_y)..(max_y + 1.0))
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(
            x.iter()
                .zip(y.iter())
                .map(|e| (*e.0, *e.1))
                .collect::<Vec<(f64, f64)>>(),
            &RED,
        ))
        .unwrap();
    }
}
