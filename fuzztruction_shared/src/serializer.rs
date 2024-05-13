pub mod instant_serializer {
    use core::time::Duration;
    use std::time::Instant;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = instant.elapsed();
        duration.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let duration = Duration::deserialize(deserializer)?;
        let instant = Instant::now().checked_sub(duration).unwrap();
        Ok(instant)
    }
}

pub mod option_instant_serializer {
    use core::time::Duration;
    use std::time::Instant;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(instant: &Option<Instant>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match instant {
            Some(instant) => {
                let duration = instant.elapsed();
                duration.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Instant>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Duration>::deserialize(deserializer)
            .map(|duration| duration.map(|d| Instant::now().checked_sub(d).unwrap()))
    }
}
