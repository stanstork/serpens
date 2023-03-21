use std::ops::{Add, Div, Mul};

use crate::{
    generator::Generator,
    matrix::{matrix::Matrix, size::Size},
    num::Num,
};

use super::shape::Shape;

pub struct Vector<T> {
    elements: Vec<T>,
    shape: Shape,
}

impl<T> Vector<T>
where
    T: Mul<Output = T> + Add<Output = T> + Num,
{
    pub fn new(elements: Vec<T>, shape: Shape) -> Self {
        Self { elements, shape }
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }

    pub fn transpose(&mut self) {
        self.shape = if self.shape == Shape::Row {
            Shape::Col
        } else {
            Shape::Row
        }
    }

    pub fn random(low: T, high: T, size: usize, shape: Shape) -> Vector<T> {
        Vector {
            elements: Generator::random_elements(low, high, size),
            shape,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.elements.get(index)
    }

    pub fn equals(&self, other: &Vector<T>) -> bool {
        if self.size() != other.size() {
            return false;
        }

        for i in 0..self.size() {
            if self.get(i).unwrap() != other.get(i).unwrap() {
                return false;
            }
        }

        true
    }

    pub fn dot(&self, right: &Vector<T>) -> Result<T, &'static str> {
        if self.size() != right.size() {
            return Err("invalid vectors size");
        }

        let mut product: T = T::zero();
        for i in 0..self.elements.len() {
            product += self.elements[i] * right.elements[i];
        }

        Ok(product)
    }

    pub fn outer(&self, right: &Vector<T>) -> Matrix<T> {
        let mut matrix = Matrix {
            elements: vec![T::zero(); self.size() * right.size()],
            size: Size::new(self.size(), right.size()),
        };

        for row in 0..self.size() {
            for col in 0..matrix.size().cols() {
                let mut cell: T = T::zero();
                cell += *self.get(row).unwrap() * *right.get(col).unwrap();
                matrix.set(row, col, cell);
            }
        }

        matrix
    }

    pub fn sum(&self) -> T {
        self.elements.iter().fold(T::zero(), |mut a, &b| {
            a += b;
            a
        })
    }

    pub fn mean(&self) -> T
    where
        T: Div<Output = T>,
    {
        self.sum() / T::from_usize(self.elements.len())
    }

    pub fn min(&self) -> T {
        self.elements().iter().fold(T::max(), |a, &b| a.get_min(&b))
    }

    pub fn max(&self) -> T {
        self.elements().iter().fold(T::min(), |a, &b| a.get_max(&b))
    }

    pub fn mul(&self, val: T) -> Vector<T> {
        Vector {
            elements: self.map(val, |a, &b| a * b),
            shape: self.shape,
        }
    }

    pub fn add(&self, val: T) -> Vector<T> {
        Vector {
            elements: self.map(val, |a, &b| a + b),
            shape: self.shape,
        }
    }

    pub fn add_vector(&self, right: &Vector<T>) -> Result<Vector<T>, &'static str> {
        if self.size() != right.size() {
            return Err("invalid vectors size");
        }

        let mut elements: Vec<T> = vec![T::zero(); self.size()];
        for i in 0..self.elements.len() {
            elements[i] = self.elements[i] + right.elements[i];
        }

        Ok(Vector {
            elements,
            shape: self.shape,
        })
    }

    pub fn sorted(&self) -> Vector<T> {
        let mut elements = self.elements.clone();
        elements.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Vector::new(elements, *self.shape())
    }

    pub fn map(&self, val: T, f: fn(T, &T) -> T) -> Vec<T> {
        self.elements.iter().map(|e| f(val, e)).collect()
    }

    pub fn elements(&self) -> &[T] {
        self.elements.as_ref()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl<T> Mul<T> for Vector<T>
where
    T: Mul<Output = T> + Add<Output = T> + Num,
{
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let elements = self.map(rhs, |a, &b| a * b);
        Vector::new(elements, *self.shape())
    }
}

impl<T> Add<T> for Vector<T>
where
    T: Mul<Output = T> + Add<Output = T> + Num,
{
    type Output = Vector<T>;

    fn add(self, rhs: T) -> Self::Output {
        let elements = self.map(rhs, |a, &b| a + b);
        Vector::new(elements, *self.shape())
    }
}

impl<T> Clone for Vector<T>
where
    T: Num,
{
    fn clone(&self) -> Self {
        Self {
            elements: self.elements.clone(),
            shape: self.shape,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{matrix::matrix::Matrix, vector::shape::Shape};

    use super::Vector;

    static ELEMENTS: &'static [i32] = &[1, 3, 5, 2, 7];

    #[test]
    fn test_init() {
        let vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Col);

        assert_eq!(*vec.shape(), Shape::Col);
        assert_eq!(vec.elements.len(), ELEMENTS.len());
    }

    #[test]
    fn test_transpose() {
        let mut vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Col);

        assert_eq!(*vec.shape(), Shape::Col);
        vec.transpose();
        assert_eq!(*vec.shape(), Shape::Row);
    }

    #[test]
    fn test_dot() {
        let left: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let right: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Col);

        let dot: Result<i32, &str> = left.dot(&right);
        assert_eq!(dot.unwrap(), 88);

        //right.transpose();
        //dot = left.dot(&right);
        //assert_eq!(dot.unwrap_err(), "invalid vectors shape");
    }

    #[test]
    fn test_random() {
        let size: usize = 100;
        let low: i32 = 0;
        let high: i32 = 1000;
        let v: Vector<i32> = Vector::random(low, high, size, Shape::Col);

        assert_eq!(v.elements().len(), size);
        assert_eq!(*v.shape(), Shape::Col);

        for i in 0..size {
            if v.elements[i] < low || v.elements[i] >= high {
                panic!("{}", "invalid vector element")
            }
        }

        let low: f64 = 1.0;
        let high: f64 = 10.0;
        let v: Vector<f64> = Vector::random(low, high, size, Shape::Row);

        assert_eq!(v.elements().len(), size);
        assert_eq!(*v.shape(), Shape::Row);
    }

    #[test]
    fn test_mul() {
        let vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let res: Vector<i32> = vec.mul(2);
        for i in 0..res.size() {
            assert_eq!(*res.get(i).unwrap(), ELEMENTS[i] * 2);
        }

        let mul_vec: Vector<i32> = vec * 3;
        for i in 0..mul_vec.size() {
            assert_eq!(*mul_vec.get(i).unwrap(), ELEMENTS[i] * 3);
        }
    }

    #[test]
    fn test_add() {
        let vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let res: Vector<i32> = vec.add(2);
        for i in 0..res.size() {
            assert_eq!(*res.get(i).unwrap(), ELEMENTS[i] + 2);
        }

        let mul_vec: Vector<i32> = vec + 3;
        for i in 0..mul_vec.size() {
            assert_eq!(*mul_vec.get(i).unwrap(), ELEMENTS[i] + 3);
        }
    }

    #[test]
    fn test_outer() {
        let left: Vector<i32> = Vector::new(vec![3, 2, 1], Shape::Col);
        let right: Vector<i32> = Vector::new(vec![7, 2, 3, 1], Shape::Row);
        let outer_p: Matrix<i32> = left.outer(&right);
        let expected: Matrix<i32> = Matrix::new(&vec![
            vec![21, 6, 9, 3],
            vec![14, 4, 6, 2],
            vec![7, 2, 3, 1],
        ]);

        assert!(outer_p.equals(&expected));
    }

    #[test]
    fn test_equals() {
        let vec1: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let vec2: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let vec3: Vector<i32> = Vector::new(vec![3, 2, 1], Shape::Col);

        assert!(vec1.equals(&vec2));
        assert!(!vec1.equals(&vec3));
    }

    #[test]
    fn test_add_vector() {
        let vec1: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let vec2: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let vec3: Result<Vector<i32>, &str> = vec1.add_vector(&vec2);

        match vec3 {
            Ok(res) => {
                for i in 0..res.size() {
                    assert_eq!(*res.get(i).unwrap(), ELEMENTS[i] * 2)
                }
            }
            Err(e) => panic!("error: {}", e),
        }
    }

    #[test]
    fn test_min() {
        let vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let min: i32 = vec.min();
        assert_eq!(1, min);
    }

    #[test]
    fn test_max() {
        let vec: Vector<i32> = Vector::new(ELEMENTS.to_vec(), Shape::Row);
        let max: i32 = vec.max();
        assert_eq!(7, max);
    }
}
