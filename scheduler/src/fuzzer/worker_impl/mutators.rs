use fuzztruction_shared::mutation_cache_entry::MutationCacheEntry;
use rand::{self, Rng};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, cmp, fmt, rc::Rc, sync::Arc, time::Duration};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MutatorType {
    Nop,
    FlipBit,
    FlipByte,
    RandomByte(usize),
    U8Counter,
    Havoc,
    Combine,
}

/// A Mutator is a collection of multiple mutations that are consecutively
/// applied, while the previously applied mutation is reverted first.
/// Thus, each iteration must revert the mutations from the previous iteration
/// and in case the mutator is dropped, the last applied mutation must also be
/// reverted.
pub trait Mutator<'a>: Iterator + fmt::Debug {
    fn mutator_type(&self) -> MutatorType;

    /// This Mutator needs the source to be synchronized before it is
    /// executed the first time.
    fn needs_sync(&self) -> bool {
        false
    }

    /// The [MutationCacheEntry] that is mutated by the [Mutator] must be enabled
    /// before the first execution and disabled after dropping the [Mutator].
    fn one_shot(&self) -> bool {
        false
    }

    fn estimate_runtime(&self, avg_exec_time: Duration) -> Duration {
        avg_exec_time.saturating_mul(self.steps_total().try_into().unwrap_or(u32::MAX))
    }

    /// How many different mutations does this Mutator
    /// apply in total.
    fn steps_total(&self) -> usize;

    /// How many mutations already have been tested.
    fn steps_done(&self) -> usize;

    /// Number of mutations left in this Mutator.
    fn steps_left(&self) -> usize {
        self.steps_total() - self.steps_done()
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>>;
}

// /// A Mutator that replaces a byte at a random index with a random value.
// pub struct RandomByte<'a> {
//     /// The buffer that is mutated.
//     buffer: &'a mut [u8],
//     /// Number of iterations we already performed.
//     current_step: usize,
//     /// Maximum number of iterations.
//     max_step: usize,
//     /// Index of the last byte we mutated.
//     last_idx: Option<usize>,
//     /// The original value of the byte that we mutated.
//     orig_byte: u8,
// }

// impl RandomByte<'_> {
//     pub fn new(buffer: &mut [u8], steps: usize) -> RandomByte {
//         RandomByte {
//             buffer,
//             current_step: 0,
//             max_step: steps,
//             last_idx: None,
//             orig_byte: 0,
//         }
//     }
// }

// impl fmt::Debug for RandomByte<'_> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("RandomByte")
//             .field("current_step", &self.current_step)
//             .field("max_step", &self.max_step)
//             .field("last_idx", &self.last_idx)
//             .field("orig_byte", &self.orig_byte)
//             .field("buffer.len()", &self.buffer.len())
//             .finish_non_exhaustive()
//     }
// }

// impl Mutator for RandomByte<'_> {
//     fn steps_total(&self) -> usize {
//         self.max_step
//     }

//     fn steps_done(&self) -> usize {
//         self.current_step
//     }

//     fn mutator_type(&self) -> MutatorType {
//         MutatorType::RandomByte
//     }
// }

// impl Iterator for RandomByte<'_> {
//     type Item = ();

//     fn next(&mut self) -> Option<Self::Item> {
//         // Revert previous mutation.
//         if let Some(idx) = self.last_idx.take() {
//             self.buffer[idx] = self.orig_byte;
//         }

//         if self.current_step == self.max_step {
//             // No steps left
//             return None;
//         }

//         // Mutate
//         let mut rng = rand::thread_rng();
//         self.last_idx = Some(rng.gen_range(0..self.buffer.len()));
//         self.orig_byte = self.buffer[self.last_idx.unwrap()];
//         let r8: u8 = rng.gen();
//         self.buffer[self.last_idx.unwrap()] = self.buffer[self.last_idx.unwrap()] ^ r8;

//         self.current_step += 1;
//         Some(())
//     }
// }

// impl Drop for RandomByte<'_> {
//     fn drop(&mut self) {
//         if let Some(idx) = self.last_idx.take() {
//             self.buffer[idx] = self.orig_byte;
//         }
//     }
// }

/// A Mutator that replaces a byte at a random index with a random value.
pub struct RandomByte<'a, const N: usize> {
    /// The buffer that is mutated.
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    /// Number of iterations we already performed.
    current_step: usize,
    /// Maximum number of iterations.
    max_step: usize,
    /// Index of the last byte we mutated.
    last_idx: Option<usize>,
    /// The original value of the byte that we mutated.
    saved_bytes: [u8; N],
}

impl<const N: usize> RandomByte<'_, N> {
    /// Create a new mutator. Returns None, if the backing buffer is smaller than N.
    pub fn new(mce: Rc<RefCell<&mut MutationCacheEntry>>, steps: usize) -> Option<RandomByte<N>> {
        if mce.borrow().get_msk_as_slice().len() < N {
            return None;
        }

        Some(RandomByte {
            mce,
            current_step: 0,
            max_step: steps,
            last_idx: None,
            saved_bytes: [0; N],
        })
    }
}

impl<const N: usize> fmt::Debug for RandomByte<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(&format!("RandomByte{}", N))
            .field("current_step", &self.current_step)
            .field("max_step", &self.max_step)
            .field("last_idx", &self.last_idx)
            .field("saved_bytes", &self.saved_bytes)
            .field("buffer.len()", &self.mce.borrow().get_msk_as_slice().len())
            .finish_non_exhaustive()
    }
}

impl<'a, const N: usize> Mutator<'a> for RandomByte<'a, N> {
    fn steps_total(&self) -> usize {
        self.max_step
    }

    fn steps_done(&self) -> usize {
        self.current_step
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::RandomByte(N)
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl<const N: usize> Iterator for RandomByte<'_, N> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        // Revert previous mutation.
        if let Some(idx) = self.last_idx.take() {
            mask[idx..(idx + N)].copy_from_slice(&self.saved_bytes);
        }

        if self.current_step == self.max_step {
            // No steps left
            return None;
        }

        // Mutate
        let mut rng = rand::thread_rng();
        let last_idx = rng.gen_range(0..=(mask.len() - N));
        self.saved_bytes
            .copy_from_slice(&mask[last_idx..(last_idx + N)]);

        mask[last_idx..(last_idx + N)]
            .iter_mut()
            .for_each(|e| *e ^= rng.gen::<u8>());
        self.last_idx = Some(last_idx);

        self.current_step += 1;
        Some(())
    }
}

impl<const N: usize> Drop for RandomByte<'_, N> {
    fn drop(&mut self) {
        if let Some(idx) = self.last_idx.take() {
            self.mce.borrow().get_msk_as_slice()[idx..(idx + N)].copy_from_slice(&self.saved_bytes);
        }
    }
}

pub type RandomByte1<'a> = RandomByte<'a, 1>;
#[allow(unused)]
pub type RandomByte2<'a> = RandomByte<'a, 2>;
pub type RandomByte4<'a> = RandomByte<'a, 4>;

/// A Mutator that consecutively flips each byte.
pub struct FlipByte<'a> {
    /// The mutated buffer.
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    /// The next byte index that is mutated.
    next_idx: usize,
    /// Index of the last byte we mutated.
    last_idx: Option<usize>,
    /// The original value of the byte that we mutated.
    orig_byte: u8,
}

impl FlipByte<'_> {
    #[allow(unused)]
    pub fn new(mce: Rc<RefCell<&mut MutationCacheEntry>>) -> FlipByte {
        let size = mce.borrow().get_msk_as_slice().len();

        FlipByte {
            mce,
            next_idx: 0,
            last_idx: None,
            orig_byte: 0,
        }
    }
}

impl fmt::Debug for FlipByte<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlipByte")
            .field("next_idx", &self.next_idx)
            .field("last_idx", &self.last_idx)
            .field("orig_byte", &self.orig_byte)
            .field("buffer.len()", &self.mce.borrow().get_msk_as_slice().len())
            .finish_non_exhaustive()
    }
}

impl<'a> Mutator<'a> for FlipByte<'a> {
    fn steps_total(&self) -> usize {
        self.mce.borrow().get_msk_as_slice().len()
    }

    fn steps_done(&self) -> usize {
        self.next_idx
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::FlipByte
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl Iterator for FlipByte<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        // Revert previous mutation.
        if let Some(idx) = self.last_idx.take() {
            mask[idx] = self.orig_byte;
        }

        if self.next_idx == mask.len() {
            // No bytes left
            return None;
        }

        // Mutate
        self.orig_byte = mask[self.next_idx];
        mask[self.next_idx] ^= 0xff;
        self.last_idx = Some(self.next_idx);

        self.next_idx += 1;

        Some(())
    }
}

impl Drop for FlipByte<'_> {
    fn drop(&mut self) {
        if let Some(idx) = self.last_idx.take() {
            self.mce.borrow().get_msk_as_slice()[idx] = self.orig_byte;
        }
    }
}

/// A Mutator  that consecutively flips each bit.
pub struct FlipBit<'a> {
    /// The buffer we are mutating.
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    /// Size of `buffer` in bits.
    size_in_bits: usize,
    /// Index of the next bit that is mutated.
    next_bit: usize,
    /// Index of the last byte we mutated.
    last_byte_idx: Option<usize>,
    /// The original value of the byte that we mutated.
    last_byte_orig_value: u8,
}

impl FlipBit<'_> {
    #[allow(unused)]
    pub fn new(mce: Rc<RefCell<&mut MutationCacheEntry>>) -> FlipBit {
        let size_in_bits = mce.borrow().get_msk_as_slice().len() * 8;

        FlipBit {
            mce,
            next_bit: 0,
            size_in_bits,
            last_byte_idx: None,
            last_byte_orig_value: 0,
        }
    }
}

impl fmt::Debug for FlipBit<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlipBit")
            .field("size_in_bits", &self.size_in_bits)
            .field("next_bit", &self.next_bit)
            .field("last_byte_idx", &self.last_byte_idx)
            .field("last_byte_orig_value", &self.last_byte_orig_value)
            .field("buffer.len()", &self.mce.borrow().get_msk_as_slice().len())
            .finish_non_exhaustive()
    }
}

impl<'a> Mutator<'a> for FlipBit<'a> {
    fn steps_total(&self) -> usize {
        self.size_in_bits
    }

    fn steps_done(&self) -> usize {
        self.next_bit
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::FlipBit
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl Iterator for FlipBit<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        // Revert previous mutation.
        if let Some(idx) = self.last_byte_idx.take() {
            mask[idx] = self.last_byte_orig_value;
        }

        if self.next_bit == self.size_in_bits {
            // No bits left
            return None;
        }

        // Store original value.
        self.last_byte_orig_value = mask[self.next_bit / 8];
        self.last_byte_idx = Some(self.next_bit / 8);

        // Mutate
        mask[self.next_bit / 8] ^= 1 << (self.next_bit % 8);

        self.next_bit += 1;

        Some(())
    }
}

impl Drop for FlipBit<'_> {
    fn drop(&mut self) {
        if let Some(idx) = self.last_byte_idx.take() {
            self.mce.borrow().get_msk_as_slice()[idx] = self.last_byte_orig_value;
        }
    }
}

/// A Mutator  that consecutively flips each bit.
pub struct U8Counter<'a> {
    /// The buffer we are mutating.
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    next_byte: usize,
    /// Index of the last byte we mutated.
    last_byte_idx: Option<usize>,
    /// The original value of the byte that we mutated.
    last_byte_orig_value: u8,
    ctr: u8,
}

impl U8Counter<'_> {
    #[allow(unused)]
    pub fn new(mce: Rc<RefCell<&mut MutationCacheEntry>>) -> U8Counter {
        let size_in_bits = mce.borrow().get_msk_as_slice().len() * 8;

        U8Counter {
            mce,
            next_byte: 0,
            last_byte_idx: None,
            last_byte_orig_value: 0,
            ctr: 0,
        }
    }
}

impl fmt::Debug for U8Counter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("U8Counter")
            .field("next_byte", &self.next_byte)
            .field("last_byte_idx", &self.last_byte_idx)
            .field("last_byte_orig_value", &self.last_byte_orig_value)
            .field("ctr", &self.ctr)
            .field("buffer.len()", &self.mce.borrow().get_msk_as_slice().len())
            .finish_non_exhaustive()
    }
}

impl<'a> Mutator<'a> for U8Counter<'a> {
    fn steps_total(&self) -> usize {
        self.mce.borrow().get_msk_as_slice().len() * 2usize.pow(8)
    }

    fn steps_done(&self) -> usize {
        self.next_byte
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::U8Counter
    }

      fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl Iterator for U8Counter<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        // Revert previous mutation.
        if let Some(idx) = self.last_byte_idx.take() {
            mask[idx] = self.last_byte_orig_value;
        }

        if self.next_byte == mask.len() {
            return None;
        }

        // Store original value.
        self.last_byte_orig_value = mask[self.next_byte];
        self.last_byte_idx = Some(self.next_byte);

        // Mutate
        mask[self.next_byte] ^= self.ctr;
        match self.ctr.checked_add(1) {
            Some(v) => self.ctr = v,
            _ => {
                self.ctr = 0;
                self.next_byte += 1;
            }
        }

        Some(())
    }
}

impl Drop for U8Counter<'_> {
    fn drop(&mut self) {
        if let Some(idx) = self.last_byte_idx.take() {
            self.mce.borrow().get_msk_as_slice()[idx] = self.last_byte_orig_value;
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, PartialOrd)]
enum HavocMutationType {
    FlipBit = 0,
    FlipByte,
    RandomByte,
    InterestingValue,
}

impl From<u8> for HavocMutationType {
    fn from(value: u8) -> Self {
        match value {
            0 => HavocMutationType::FlipBit,
            1 => HavocMutationType::FlipByte,
            2 => HavocMutationType::RandomByte,
            3 => HavocMutationType::InterestingValue,
            _ => unreachable!(),
        }
    }
}

const INTERESTING_VALUES: [u8; 4] = [0x80, 0x1, 0x0, 0x2];

/// A Mutator that (pseudo)-randomly applies `max_stacks` mutations
/// until `max_steps` steps have been performed
pub struct Havoc<'a> {
    /// The buffer we are mutating.
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    /// Original buffer value
    orig_buffer: Vec<u8>,
    steps_done: usize,
    // reset stacked mutations after `max_stacks` iterations
    max_stacks: usize,
    // the total number of iterations before iterator exhaustion
    repetitions: usize,
    steps_total: usize,
}

impl Havoc<'_> {
    #[allow(unused)]
    pub fn new(
        mce: Rc<RefCell<&mut MutationCacheEntry>>,
        max_stacks: usize,
        repetitions: usize,
    ) -> Havoc {
        // save un-mutated buffer
        // TODO: replace Vec?
        let mut orig_buffer: Vec<u8> = vec![0; mce.borrow().get_msk_as_slice().len()];
        orig_buffer.copy_from_slice(mce.borrow().get_msk_as_slice());
        let steps_total = max_stacks * repetitions;
        Havoc {
            mce,
            orig_buffer,
            steps_done: 0,
            steps_total,
            max_stacks,
            repetitions,
        }
    }
}

impl fmt::Debug for Havoc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Havoc")
            .field("steps_done", &self.steps_done)
            .field("max_stacks", &self.max_stacks)
            .field("repetitions", &self.repetitions)
            .field("orig_buffer.len()", &self.orig_buffer.len())
            .field("buffer.len()", &self.mce.borrow().get_msk_as_slice().len())
            .finish_non_exhaustive()
    }
}

impl<'a> Mutator<'a> for Havoc<'a> {
    fn steps_total(&self) -> usize {
        self.steps_total
    }

    fn steps_done(&self) -> usize {
        self.steps_done as usize
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::Havoc
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl Iterator for Havoc<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        // check if we exhausted the iterator
        if self.steps_done == self.steps_total() {
            mask.copy_from_slice(&self.orig_buffer);
            return None;
        }
        // revert previous mutations
        if self.steps_done > 0 && self.steps_done % self.max_stacks == 0 {
            mask.copy_from_slice(&self.orig_buffer);
        }
        // choose random mutation, apply & yield
        let mut rng = rand::thread_rng();
        let mutation_choice: u8 = rng.gen::<u8>() % 4;
        let random_idx = rng.gen::<usize>();

        match mutation_choice.into() {
            HavocMutationType::FlipBit => {
                let idx = (random_idx / 8) % mask.len();
                mask[idx] ^= 1 << (random_idx % 8);
            }
            HavocMutationType::FlipByte => {
                mask[random_idx % mask.len()] ^= 0xff;
            }
            HavocMutationType::RandomByte => {
                mask[random_idx % mask.len()] = rng.gen::<u8>();
            }
            HavocMutationType::InterestingValue => {
                // TODO: optimize this: only replace first/last byte?
                let idx = random_idx % INTERESTING_VALUES.len();
                for b in mask.iter_mut() {
                    *b = INTERESTING_VALUES[idx];
                }
            }
        }
        // increment mutation counter
        self.steps_done += 1;
        Some(())
    }
}

impl Drop for Havoc<'_> {
    fn drop(&mut self) {
        self.mce
            .borrow()
            .get_msk_as_slice()
            .copy_from_slice(&self.orig_buffer);
    }
}

pub struct Nop {
    /// Number of times this Mutator yields.
    steps: usize,
    steps_done: usize,
}

impl Nop {
    #[allow(unused)]
    pub fn new(steps: usize) -> Nop {
        Nop {
            steps,
            steps_done: 0,
        }
    }
}

impl fmt::Debug for Nop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nop")
            .field("steps", &self.steps)
            .field("steps_done", &self.steps_done)
            .finish()
    }
}

impl<'a> Mutator<'a> for Nop {
    fn steps_total(&self) -> usize {
        self.steps
    }

    fn steps_done(&self) -> usize {
        self.steps_done
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::Nop
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        unimplemented!()
    }
}

impl Iterator for Nop {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        if self.steps == 0 {
            None
        } else {
            self.steps -= 1;
            Some(())
        }
    }
}

pub struct CombineMutator<'a> {
    /// Number of times this Mutator yields.
    steps: usize,
    steps_done: usize,
    msks: Vec<Arc<[u8]>>,
    mce: Rc<RefCell<&'a mut MutationCacheEntry>>,
    buffer_original: Box<[u8]>,
    one_shot: bool,
}

impl CombineMutator<'_> {
    #[allow(unused)]
    pub fn new(
        mce: Rc<RefCell<&mut MutationCacheEntry>>,
        msks: Vec<Arc<[u8]>>,
        one_shot: bool,
    ) -> CombineMutator {
        let buffer_original = mce.borrow().get_msk_as_slice().to_vec().into_boxed_slice();

        CombineMutator {
            steps: msks.len(),
            msks,
            steps_done: 0,
            mce,
            buffer_original,
            one_shot,
        }
    }
}

impl fmt::Debug for CombineMutator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CombineMutator")
            .field("steps", &self.steps)
            .field("steps_done", &self.steps_done)
            .field("msks.len()", &self.msks.len())
            .finish()
    }
}

impl<'a> Mutator<'a> for CombineMutator<'a> {
    fn steps_total(&self) -> usize {
        self.steps
    }

    fn steps_done(&self) -> usize {
        self.steps_done
    }

    fn mutator_type(&self) -> MutatorType {
        MutatorType::Combine
    }

    fn one_shot(&self) -> bool {
        self.one_shot
    }

    fn needs_sync(&self) -> bool {
        true
    }

    fn target(&self) -> Rc<RefCell<&'a mut MutationCacheEntry>> {
        self.mce.clone()
    }
}

impl Iterator for CombineMutator<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let mce = self.mce.borrow();
        let mask = mce.get_msk_as_slice();
        if let Some(msk) = self.msks.pop() {
            self.steps_done += 1;
            // The msk buffer might be smaller then ours, thus we return the original value.
            mask.copy_from_slice(&self.buffer_original[..]);
            let copy_len = cmp::min(mask.len(), msk.len());
            mask[0..copy_len].copy_from_slice(&msk[0..copy_len]);
            Some(())
        } else {
            None
        }
    }
}

impl Drop for CombineMutator<'_> {
    fn drop(&mut self) {
        self.mce
            .borrow()
            .get_msk_as_slice()
            .copy_from_slice(&self.buffer_original[..]);
    }
}

#[cfg(test)]
mod test {
    use std::mem;

    use crate::fuzzer::worker_impl::mutators::Havoc;

    use super::RandomByte1;

    #[test]
    fn havoc_mutator_drop() {
        const NUM_STEPS: usize = 100;
        let mut buffer = vec![0; 64];
        let orig_buffer: &[u8] = &[0; 64];
        // TODO: test more cases; add seed
        // Test reset after exhaustion
        let mut ctr = 0;
        let num_resets = 3;
        {
            let mut mutator = Havoc::new(&mut buffer, NUM_STEPS, num_resets * NUM_STEPS);
            for _ in 0..num_resets {
                for _ in 0..NUM_STEPS {
                    mutator.next();
                    ctr += 1;
                }
            }
        }
        // assert that the final buffer is reset to the initial
        assert_eq!(orig_buffer, buffer);
        // assert that num_resets * NUM_STEPS mutations were conducted
        assert_eq!(ctr, num_resets * NUM_STEPS);
    }

    #[test]
    fn test_random_mutator_1() {
        let mut buffer = vec![11u8; 7];
        let mutator = RandomByte1::new(&mut buffer, 100).unwrap();
        // for (idx, _) in mutator.enumerate() {
        //     // pass
        //     if idx == 50 {
        //         break;
        //     }
        // }
        mem::drop(mutator);
        assert_eq!(buffer, vec![11u8; 7]);
    }
}
