use std::ops::{Add, Mul};

use crate::{matrix::matrix::Matrix, num::Num};

pub struct PrintHelper {}

impl PrintHelper {
    pub fn print_col<T>(elements: &[T], max: T)
    where
        T: Num,
    {
        if elements.is_empty() {
            println!("[]");
            return;
        }

        let width: usize = max.to_string().len();

        let first = elements.first().unwrap();
        println!("⎡ {first:>width$} ⎤");

        for element in elements.iter().take(elements.len() - 1).skip(1) {
            println!("⎢ {element:>width$} ⎢");
        }

        let last = elements.last().unwrap();
        println!("⎣ {last:>width$} ⎦");
    }

    pub fn print_rows<T: Mul<Output = T> + Add<Output = T> + Num>(matrix: &Matrix<T>, max: T) {
        let width: usize = max.to_string().len();
        for r in 0..matrix.size().rows() {
            let row: Vec<T> = matrix.get_row(r);

            let first = row.first().unwrap();
            if r == 0 {
                print!("⎡ {first:>width$} ");
            } else if r == matrix.size().rows() - 1 {
                print!("⎣ {first:>width$} ");
            } else {
                print!("⎢ {first:>width$} ");
            }

            for element in row.iter().take(row.len() - 1).skip(1) {
                print!(" {element:>width$} ");
            }

            let last = row.last().unwrap();
            if r == 0 {
                println!(" {last:>width$} ⎤");
            } else if r == matrix.size().rows() - 1 {
                println!(" {last:>width$} ⎦");
            } else {
                println!(" {last:>width$} ⎥");
            }
        }
    }
}
