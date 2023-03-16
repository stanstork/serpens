use std::{
    fs::File,
    io::{BufRead, BufReader, Error, Lines},
    ops::{Add, Mul},
};

use linear::{
    num::Num,
    vector::{shape::Shape, vector::Vector},
};

pub struct Reader {}

impl Reader {
    pub fn read_2d_csv<T>(path: &str) -> Result<(Vector<T>, Vector<T>), std::io::Error>
    where
        T: Mul<Output = T> + Add<Output = T> + Num,
    {
        let lines: Result<Lines<BufReader<File>>, Error> = Self::get_lines(path);
        match lines {
            Ok(lines) => {
                let mut x: Vec<T> = vec![];
                let mut y: Vec<T> = vec![];

                for line in lines {
                    match line {
                        Ok(line) => {
                            let data: Vec<&str> = line.split(',').collect();

                            x.push(T::from_str(data[0]));
                            y.push(T::from_str(data[1]));
                        }
                        Err(e) => println!("ERROR: {}", e),
                    }
                }

                return Ok((Vector::new(x, Shape::Col), Vector::new(y, Shape::Col)));
            }
            Err(e) => return Err(e),
        }
    }

    fn get_lines(path: &str) -> Result<Lines<BufReader<File>>, Error> {
        let file: File = File::open(path)?;
        let reader: BufReader<File> = BufReader::new(file);

        Ok(reader.lines())
    }
}

#[cfg(test)]
mod test {
    use std::io::Error;

    use linear::vector::vector::Vector;

    use super::Reader;

    #[test]
    fn test_read_2d_csv() {
        let data: Result<(Vector<f64>, Vector<f64>), Error> =
            Reader::read_2d_csv("test_data/data_1d.csv");

        match data {
            Ok(data) => {
                assert_eq!(data.0.size(), 100);
                assert_eq!(data.1.size(), 100);

                assert_eq!(*data.0.get(0).unwrap(), 95.724162408);
                assert_eq!(*data.1.get(0).unwrap(), 197.179636092);

                assert_eq!(*data.0.get(99).unwrap(), 40.3252822158);
                assert_eq!(*data.1.get(99).unwrap(), 86.7236853299);
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
