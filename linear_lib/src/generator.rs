use rand::{distributions::uniform::SampleUniform, rngs::ThreadRng, Rng};

pub struct Generator {}

impl Generator {
    pub fn random_elements<T>(low: T, high: T, size: usize) -> Vec<T>
    where
        T: Copy + PartialOrd + SampleUniform,
    {
        let mut rng: ThreadRng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(low..high)).collect()
    }

    pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
        let step: f64 = (stop - start) / (num - 1) as f64;
        let mut space: Vec<f64> = vec![0.0; num];

        space[0] = start;

        for i in 1..num {
            space[i] = space[i - 1] + step;
        }

        space
    }
}

#[cfg(test)]
mod test {
    use super::Generator;

    #[test]
    fn test_linspace() {
        let space: Vec<f64> = Generator::linspace(2.0, 3.0, 5);
        let expected: Vec<f64> = vec![2.0, 2.25, 2.5, 2.75, 3.0];
        for i in 0..space.len() {
            assert_eq!(space[i], expected[i]);  
        }

        println!("{:?}", space);
    }
}
