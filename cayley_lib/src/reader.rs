use std::{
    fs::File,
    io::{BufRead, BufReader, Error, Lines},
    ops::{Add, Mul},
};

use linear::num::Num;

pub struct Reader {}

impl Reader {
    pub fn read_nd_csv<T>(path: &str, columns: usize) -> Result<Vec<Vec<T>>, Error>
    where
        T: Mul<Output = T> + Add<Output = T> + Num,
    {
        let lines: Result<Lines<BufReader<File>>, Error> = Self::get_lines(path);
        match lines {
            Ok(lines) => {
                let mut containers: Vec<Vec<T>> = vec![vec![]; columns];
                for line in lines {
                    match line {
                        Ok(line) => {
                            let data: Vec<&str> = line.split(',').collect();
                            for c in 0..columns {
                                containers[c].push(T::from_str(data[c]));
                            }
                        }
                        Err(e) => println!("ERROR: {}", e),
                    }
                }

                Ok(containers)
            }
            Err(e) => Err(e),
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

    use super::Reader;

    #[test]
    fn test_read_2d_csv() {
        let data: Result<Vec<Vec<f64>>, Error> = Reader::read_nd_csv("test_data/data_1d.csv", 2);

        match data {
            Ok(data) => {
                assert_eq!(data[0].len(), 100);
                assert_eq!(data[1].len(), 100);

                assert_eq!(*data[0].get(0).unwrap(), 95.724162408);
                assert_eq!(*data[1].get(0).unwrap(), 197.179636092);

                assert_eq!(*data[0].get(99).unwrap(), 40.3252822158);
                assert_eq!(*data[1].get(99).unwrap(), 86.7236853299);
            }
            Err(e) => panic!("ERROR: {}", e),
        }
    }
}
