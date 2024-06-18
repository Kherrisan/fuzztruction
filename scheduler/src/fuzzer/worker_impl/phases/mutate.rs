use std::{cell::RefCell, rc::Rc};

use super::FuzzingPhase;
use crate::fuzzer::{
    worker::FuzzingWorker,
    worker_impl::mutators::{self, Mutator},
};

use anyhow::Result;
use fuzztruction_shared::mutation_cache_entry::MutationCacheEntry;

const PHASE: FuzzingPhase = FuzzingPhase::Mutate;

impl FuzzingWorker {
    pub fn do_mutate_phase(&mut self) -> Result<()> {
        self.state.set_phase(PHASE);
        let entry = self.state.entry();

        // TODO: Update some stats to keep track whether this entry got air time.
        let qe_stats_rw = entry.stats_rw();
        drop(qe_stats_rw);

        self.load_queue_entry_mutations(&entry)?;

        let source = self.source.as_mut().unwrap();
        let candidates = source.mutation_cache().borrow_mut().entries_mut_static();

        let mut mutations = Vec::<(
            Rc<RefCell<&mut MutationCacheEntry>>,
            Vec<Box<dyn mutators::Mutator<Item = ()>>>,
        )>::new();

        for candidate in candidates.into_iter() {
            let mut mutators = Vec::new();
            let msk_len = candidate.get_msk_as_slice().len();

            let iterations = match msk_len {
                x if x <= 32 => 128 * x,
                x if x <= 128 => 64 * x,
                _ => 64 * 128,
            };

            let candidate = Rc::new(RefCell::new(candidate));

            if msk_len <= 4 {
                let mutator = mutators::U8Counter::new(candidate.clone());
                if entry.stats_rw().mark_mutator_done(mutator.mutator_type()) {
                    mutators.push(Box::new(mutator) as Box<dyn mutators::Mutator<Item = ()>>);
                }
            }

            let mutator = mutators::Havoc::new(candidate.clone(), 16, 100);
            mutators.push(Box::new(mutator) as Box<dyn mutators::Mutator<Item = ()>>);

            let mutator = mutators::RandomByte1::new(candidate.clone(), iterations);
            if let Some(mutator) = mutator {
                mutators.push(Box::new(mutator) as Box<dyn mutators::Mutator<Item = ()>>);
            }

            let mutator = mutators::RandomByte4::new(candidate.clone(), iterations);
            if let Some(mutator) = mutator {
                mutators.push(Box::new(mutator) as Box<dyn mutators::Mutator<Item = ()>>);
            }

            let mutator = mutators::FlipBit::new(candidate.clone());
            if entry.stats_rw().mark_mutator_done(mutator.mutator_type()) {
                mutators.push(Box::new(mutator) as Box<dyn mutators::Mutator<Item = ()>>);
            }

            let entry = (candidate, mutators);
            mutations.push(entry);
        }

        let cov_timeout = self.config.phases.mutate.entry_cov_timeout;
        self.fuzz_candidates(mutations, Some(cov_timeout), false)?;

        Ok(())
    }
}
