use std::ops::{Add, Div, DivAssign, Mul, SubAssign};

use linear::{
    matrix::matrix::Matrix,
    num::Num,
    vector::{shape::Shape, vector::Vector},
};

pub struct GaussianElimination {}

impl GaussianElimination {
    // Gaussian elimination with partial pivoting
    pub fn solve<T>(a: Matrix<T>, b: Vector<T>) -> Option<Vector<T>>
    where
        T: Mul<Output = T> + Add<Output = T> + Div<Output = T> + SubAssign + DivAssign + Num,
    {
        // make augmented matrix
        let mut augmented_matrix: Vec<Vec<T>> = vec![vec![T::zero(); b.size() + 1]; b.size()];
        for (r, item) in augmented_matrix
            .iter_mut()
            .enumerate()
            .take(a.size().rows())
        {
            let mut row: Vec<T> = a.get_row(r).iter().map(|e| *(*e)).collect();
            row.push(*b.get(r).unwrap());
            *item = row;
        }

        // WP algorithm from Gaussian elimination page
        // produces row-eschelon form
        for k in 0..augmented_matrix.len() {
            // Find pivot for column k:
            let mut i_max: usize = k;
            let mut max: T = augmented_matrix[k][k].absolute();

            for (i, item) in augmented_matrix
                .iter()
                .enumerate()
                .take(b.size())
                .skip(k + 1)
            {
                let abs: T = item[k].absolute();
                if abs > max {
                    i_max = i;
                    max = abs;
                }
            }

            if augmented_matrix[i_max][k] == T::zero() {
                return None;
            }

            // swap rows(k, i_max)
            let tmp: Vec<T> = augmented_matrix[k].clone();
            augmented_matrix[k] = augmented_matrix[i_max].clone();
            augmented_matrix[i_max] = tmp;

            // Do for all rows below pivot:
            for i in (k + 1)..b.size() {
                // Do for all remaining elements in current row:
                for j in (k + 1)..(b.size() + 1) {
                    let mut val: T = augmented_matrix[i][j];
                    val -=
                        augmented_matrix[k][j] * (augmented_matrix[i][k] / augmented_matrix[k][k]);
                    augmented_matrix[i][j] = val;
                }

                // Fill lower triangular matrix with zeros:
                augmented_matrix[i][k] = T::zero();
            }
        }

        // end of WP algorithm.
        // now back substitute to get result.
        let mut x: Vec<T> = vec![T::zero(); b.size()];
        for i in (0..b.size()).rev() {
            x[i] = augmented_matrix[i][b.size()];
            for j in (i + 1)..b.size() {
                let mut val: T = x[i];
                val -= augmented_matrix[i][j] * x[j];
                x[i] = val;
            }
            x[i] /= augmented_matrix[i][i]
        }

        Some(Vector::new(x, Shape::Col))
    }
}

#[cfg(test)]
mod test {
    use linear::{
        matrix::matrix::Matrix,
        vector::{shape::Shape, vector::Vector},
    };

    use super::GaussianElimination;

    #[test]
    fn test_solve_1() {
        let a: Matrix<f64> = Matrix::new(&vec![
            vec![1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            vec![1.00, 0.63, 0.39, 0.25, 0.16, 0.10],
            vec![1.00, 1.26, 1.58, 1.98, 2.49, 3.13],
            vec![1.00, 1.88, 3.55, 6.70, 12.62, 23.80],
            vec![1.00, 2.51, 6.32, 15.88, 39.90, 100.28],
            vec![1.00, 3.14, 9.87, 31.01, 97.41, 306.02],
        ]);
        let b: Vector<f64> = Vector::new(vec![-0.01, 0.61, 0.91, 0.99, 0.60, 0.02], Shape::Col);
        let x: Vector<f64> = GaussianElimination::solve(a, b).unwrap();
        let expected: Vector<f64> = Vector::new(
            vec![
                -0.01,
                1.602790394502114,
                -1.6132030599055613,
                1.2454941213714368,
                -0.4909897195846576,
                0.065760696175232,
            ],
            Shape::Col,
        );

        for i in 0..x.size() {
            assert!(expected.get(i).unwrap() - x.get(i).unwrap() < 0.0000001);
        }
    }

    #[test]
    fn test_solve_2() {
        let a: Matrix<f64> = Matrix::new(&vec![
            vec![1.0, 3.0, -2.0],
            vec![3.0, 5.0, 6.0],
            vec![2.0, 4.0, 3.0],
        ]);
        let b: Vector<f64> = Vector::new(vec![5.0, 7.0, 8.0], Shape::Col);
        let x = GaussianElimination::solve(a, b).unwrap();
        let expected: Vector<f64> = Vector::new(vec![-15.0, 8.0, 2.0], Shape::Col);

        for i in 0..x.size() {
            assert!(expected.get(i).unwrap() - x.get(i).unwrap() < f32::EPSILON as f64);
        }
    }

    #[test]
    fn test_solve_3() {
        let a: Matrix<f64> = Matrix::new(&vec![
            vec![4.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 4.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 4.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 4.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 4.0],
        ]);
        let b: Vector<f64> = Vector::new(
            vec![1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 4.0 / 5.0, 5.0 / 6.0],
            Shape::Col,
        );
        let x: Vector<f64> = GaussianElimination::solve(a, b).unwrap();
        let expected: Vector<f64> = Vector::new(
            vec![
                39.0 / 400.0,
                11.0 / 100.0,
                31.0 / 240.0,
                37.0 / 300.0,
                71.0 / 400.0,
            ],
            Shape::Col,
        );

        for i in 0..x.size() {
            assert!(expected.get(i).unwrap() - x.get(i).unwrap() < f32::EPSILON as f64);
        }
    }
}
