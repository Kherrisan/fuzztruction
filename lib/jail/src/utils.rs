pub fn generate_deterministic_random(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut seed: u64 = 12345;

    for _ in 0..size {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((seed & 0xff) as u8);
    }

    data
}
