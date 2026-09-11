//! VRAM budgeting for model loads.
//!
//! On amdgpu/Vulkan there is no cgroup limit for VRAM and no graceful OOM. A model that
//! allocates past the card's capacity does not get an allocation error - the compositor
//! loses its framebuffer instead (`amdgpu pin failed`, `-12`) and the graphical session
//! dies. So capacity is something we compute up front and refuse, never something we
//! discover at runtime.
//!
//! The numbers here were measured on an RX 7600 XT (16368 MiB) on 2026-09-07.

/// Total board usage we will never exceed, desktop included.
///
/// Deliberately far below the ~15.4 GiB free at idle: the compositor allocates *new*
/// framebuffers on demand, so leaving only its resident footprint free is what kills
/// the session.
pub const CEILING_MIB: u64 = 14_400;

/// Slack for llama.cpp's compute buffers, which scale with batch rather than context.
pub const COMPUTE_BUFFER_MIB: u64 = 500;

/// Multiplier applied to computed cache size.
///
/// The arithmetic below accounts for the KV tensors themselves. Measured allocation
/// runs above that, in graph and bookkeeping overhead not worth modelling exactly.
/// Chosen so the predicted total footprint sits just above the measured one on gemma4
/// at 262144, because the error must land on the side of refusing a load that would
/// have fitted rather than accepting one that takes the desktop down.
const SAFETY_FACTOR: f64 = 1.25;

/// Element width of a quantised KV cache entry, in bytes.
///
/// The `q*` variants are not whole numbers: a block stores a scale alongside its
/// quantised values, so the effective width is slightly above the nominal bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl CacheType {
    #[must_use]
    pub fn bytes_per_element(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 1.0625,
            Self::Q4_0 => 0.5625,
        }
    }

    /// The spelling llama.cpp expects for `--cache-type-k` / `--cache-type-v`.
    #[must_use]
    pub fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }
}

impl std::str::FromStr for CacheType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            other => anyhow::bail!("unknown cache type {other:?}, expected f16, q8_0 or q4_0"),
        }
    }
}

/// A set of attention layers sharing one geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGroup {
    pub layers: u32,
    pub kv_heads: u32,
    pub key_length: u32,
    pub value_length: u32,
}

impl LayerGroup {
    /// Bytes of cache this group needs per token of context it holds.
    #[must_use]
    pub fn bytes_per_token(&self, cache: CacheType) -> f64 {
        f64::from(self.layers)
            * f64::from(self.kv_heads)
            * f64::from(self.key_length + self.value_length)
            * cache.bytes_per_element()
    }
}

/// How a model's KV cache grows with context.
///
/// Splitting global from sliding layers is not a detail - it is the difference between
/// a 12B holding 256k tokens and a 27B holding 19k. Treating every layer as global
/// over-estimates gemma4 by 40x and would wrongly report it unable to run long context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    /// Layers whose cache grows with the full context.
    pub global: LayerGroup,
    /// Layers capped at a fixed window, paired with that window size.
    pub sliding: Option<(LayerGroup, u32)>,
}

impl KvLayout {
    /// A model where every layer attends over the whole context.
    #[must_use]
    pub fn dense(layers: u32, kv_heads: u32, key_length: u32, value_length: u32) -> Self {
        Self {
            global: LayerGroup {
                layers,
                kv_heads,
                key_length,
                value_length,
            },
            sliding: None,
        }
    }

    /// Marginal cost of one more token of context, once past the sliding window.
    ///
    /// This is the number that actually governs how far context can be pushed, since
    /// the sliding layers stop growing.
    #[must_use]
    pub fn bytes_per_token(&self, cache: CacheType) -> f64 {
        self.global.bytes_per_token(cache) * SAFETY_FACTOR
    }

    /// Total cache bytes for `context` tokens.
    #[must_use]
    pub fn cache_bytes(&self, cache: CacheType, context: u64) -> f64 {
        let mut bytes = self.global.bytes_per_token(cache) * context as f64;
        if let Some((group, window)) = self.sliding {
            let held = context.min(u64::from(window));
            bytes += group.bytes_per_token(cache) * held as f64;
        }
        bytes * SAFETY_FACTOR
    }

    /// Total cache for `context` tokens, in MiB, rounded up.
    #[must_use]
    pub fn cache_mib(&self, cache: CacheType, context: u64) -> u64 {
        (self.cache_bytes(cache, context) / (1024.0 * 1024.0)).ceil() as u64
    }
}

/// Memory held back from a discrete GPU for the desktop.
///
/// The compositor allocates framebuffers on demand, so leaving only its resident
/// footprint free is what kills the session. Derived from measurement: 16368 MiB of
/// board memory against a ceiling of 14400 that never crashed.
const DISCRETE_RESERVE_MIB: u64 = 2048;

/// Memory held back on a unified-memory device.
///
/// A share of the total rather than a constant, because on unified memory the reserve
/// is what the whole rest of the machine lives on. llama.cpp reports macOS's
/// recommended working set as the device total, and that is already ~80% of physical
/// RAM - so the flat 1024 MiB this used to hold back left a 64 GB Mac running its
/// browser, editor and agents in the remaining 12 GB.
///
/// Measured on an M4 Max/64 GB on 2026-09-11, with a 30B loaded at its trained 256k
/// context under the flat reserve: 17 GB in swap, 14 GB held by the compressor, 67 MB
/// free. Over-committing here does degrade into paging rather than killing the
/// session - but paging is not a mild outcome for a model. macOS compresses the
/// weights while the server sits idle, so the next prompt pays to fault them back:
/// 13.4 tok/s on the first request against 97.3 tok/s once resident.
///
/// A quarter leaves the rest of that machine ~24 GiB, which is what a desktop running
/// a browser and an editor actually uses.
fn unified_reserve_mib(total_mib: u64) -> u64 {
    (total_mib / 4).max(1024)
}

/// What a load has to fit inside.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Total board or wired memory we will never exceed.
    pub ceiling_mib: u64,
    /// Memory already held by everything that is not the model being loaded.
    pub desktop_mib: u64,
    /// Largest context a launch may take, however much more would fit.
    ///
    /// Memory is not the only thing a long context costs. Attention is quadratic, so
    /// the marginal prefill rate on the machine above fell from 582 tok/s at 8k to 48
    /// tok/s at 57k - and a harness fills whatever window it is told about, so an
    /// advertised 256k becomes a real 128k session that generates at 9 tok/s and
    /// reprocesses for minutes whenever its cache is lost. The cache for a context
    /// nobody reaches is also allocated up front and then paged out by the OS.
    ///
    /// `None` means whatever fits, which is the right answer for a card small enough
    /// that memory caps the context first.
    pub context_cap: Option<u64>,
}

impl Budget {
    /// A budget for this project's reference machine.
    ///
    /// Kept for the discrete-GPU case where the ceiling is a measured constant rather
    /// than something the backend reports.
    #[must_use]
    pub fn new(desktop_mib: u64) -> Self {
        Self {
            ceiling_mib: CEILING_MIB,
            desktop_mib,
            context_cap: None,
        }
    }

    /// The same budget with a ceiling on how much context a launch may take.
    #[must_use]
    pub fn capped_at(self, context_cap: Option<u64>) -> Self {
        Self {
            context_cap,
            ..self
        }
    }

    /// A budget derived from whatever llama.cpp says the device has.
    ///
    /// The reserve depends on how the device shares memory, not on the operating
    /// system: a discrete card has to leave room for the compositor, while unified
    /// memory is already capped by the OS on our behalf.
    #[must_use]
    pub fn for_device(device: &crate::device::Device) -> Self {
        let reserve = if device.backend.is_unified_memory() {
            unified_reserve_mib(device.total_mib)
        } else {
            DISCRETE_RESERVE_MIB
        };
        Self {
            ceiling_mib: device.total_mib.saturating_sub(reserve),
            // What the device reports free already excludes everything else resident,
            // so the baseline is whatever is missing from the total.
            desktop_mib: device.total_mib.saturating_sub(device.free_mib),
            context_cap: None,
        }
    }

    /// MiB available to llama-server for weights, KV cache and compute buffers.
    ///
    /// Saturates at zero rather than underflowing when the baseline alone is over the
    /// ceiling, which would mean nothing can be loaded at all.
    #[must_use]
    pub fn available_mib(&self) -> u64 {
        self.ceiling_mib
            .saturating_sub(self.desktop_mib)
            .saturating_sub(COMPUTE_BUFFER_MIB)
    }

    /// Largest context this budget allows alongside `weights_mib`, or `None` if the
    /// weights alone do not fit.
    #[must_use]
    pub fn max_context(&self, kv: &KvLayout, cache: CacheType, weights_mib: u64) -> Option<u64> {
        let fits = self.context_that_fits(kv, cache, weights_mib)?;
        Some(self.context_cap.map_or(fits, |cap| fits.min(cap)))
    }

    /// Largest context the memory alone allows, before any cap on it.
    fn context_that_fits(&self, kv: &KvLayout, cache: CacheType, weights_mib: u64) -> Option<u64> {
        let for_cache = self.available_mib().checked_sub(weights_mib)?;
        let budget_bytes = for_cache as f64 * 1024.0 * 1024.0 / SAFETY_FACTOR;

        let global = kv.global.bytes_per_token(cache);
        let (sliding, window) = kv.sliding.map_or((0.0, 0u64), |(g, w)| {
            (g.bytes_per_token(cache), u64::from(w))
        });

        // Below the window every layer still grows, so both groups charge per token.
        let within = budget_bytes / (global + sliding);
        if within <= window as f64 {
            return Some(within as u64);
        }

        // Past it the sliding layers are a fixed cost and only the global ones grow.
        if global <= 0.0 {
            return Some(u64::MAX);
        }
        let past = (budget_bytes - sliding * window as f64) / global;
        Some(past.max(0.0) as u64)
    }

    /// Whether a specific load fits. This is the check that must gate every spawn.
    #[must_use]
    pub fn fits(&self, kv: &KvLayout, cache: CacheType, weights_mib: u64, context: u64) -> bool {
        weights_mib + kv.cache_mib(cache, context) <= self.available_mib()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen3-14b: 40 layers, 8 KV heads, 128/128. Measured 84.8 KiB/token at q8_0 by
    /// differencing VRAM between ctx=4096 and ctx=32768 on the real card.
    fn qwen3_14b() -> KvLayout {
        KvLayout::dense(40, 8, 128, 128)
    }

    /// qwen3.6-27b (arch `qwen35`): 65 layers, 4 KV heads, 256/256, global attention on
    /// every layer. Read from the first 8 MiB of the blob.
    fn qwen36_27b() -> KvLayout {
        KvLayout::dense(65, 4, 256, 256)
    }

    /// gemma4-12b: 48 blocks in a repeating pattern of 5 sliding then 1 global. The
    /// sliding layers carry 8 KV heads at 256/256 over a 1024 window; the global layers
    /// carry a single KV head at 512/512.
    fn gemma4_12b() -> KvLayout {
        KvLayout {
            global: LayerGroup {
                layers: 8,
                kv_heads: 1,
                key_length: 512,
                value_length: 512,
            },
            sliding: Some((
                LayerGroup {
                    layers: 40,
                    kv_heads: 8,
                    key_length: 256,
                    value_length: 256,
                },
                1024,
            )),
        }
    }

    #[test]
    fn kv_per_token_matches_the_measured_card() {
        // Compare raw geometry against the measurement, without the safety factor.
        let kib = qwen3_14b().global.bytes_per_token(CacheType::Q8_0) / 1024.0;
        assert!(
            (kib - 84.8).abs() < 1.0,
            "predicted {kib:.1} KiB/token, measured 84.8"
        );
    }

    #[test]
    fn q8_0_cache_is_about_half_of_f16() {
        let g = qwen3_14b().global;
        let ratio = g.bytes_per_token(CacheType::F16) / g.bytes_per_token(CacheType::Q8_0);
        assert!((ratio - 1.882).abs() < 0.01);
    }

    /// The load that actually killed the desktop: qwen3-14b at ctx=65536.
    #[test]
    fn rejects_the_load_that_crashed_the_machine() {
        let budget = Budget::new(900);
        assert!(
            !budget.fits(&qwen3_14b(), CacheType::Q8_0, 8836, 65536),
            "must refuse the ctx=65536 load that took the compositor down"
        );
        assert!(budget.fits(&qwen3_14b(), CacheType::Q8_0, 8836, 32768));
    }

    /// A 27B cannot hold a long context here at any quant, because quantising weights
    /// does not shrink the cache.
    #[test]
    fn dense_27b_cannot_reach_long_context() {
        let budget = Budget::new(900);
        let ctx = budget
            .max_context(&qwen36_27b(), CacheType::Q8_0, 10428)
            .expect("IQ3_XXS weights fit");
        assert!((14_000..18_000).contains(&ctx), "expected ~16k, got {ctx}");
        assert!(!budget.fits(&qwen36_27b(), CacheType::Q8_0, 10428, 200_000));
    }

    /// The regression that matters most: gemma4 measurably holds its full 262144
    /// context at 11242 MiB peak. Treating its sliding layers as global predicted
    /// ~15k and would have declared the daily driver unusable.
    #[test]
    fn sliding_window_model_reaches_its_full_trained_context() {
        let budget = Budget::new(900);
        let ctx = budget
            .max_context(&gemma4_12b(), CacheType::Q8_0, 7039)
            .expect("weights fit");
        assert!(
            ctx >= 262_144,
            "gemma4 holds 262144 on the real card, predicted only {ctx}"
        );
        assert!(budget.fits(&gemma4_12b(), CacheType::Q8_0, 7039, 262_144));
    }

    /// The estimate has to match reality, and specifically must not come in under it.
    ///
    /// Measured on the card: gemma4-12b at ctx=262144 peaked 10368 MiB above the
    /// desktop baseline, covering weights, KV cache and compute buffers.
    #[test]
    fn predicted_footprint_covers_the_measured_one() {
        const MEASURED_MIB: u64 = 10_368;
        const WEIGHTS_MIB: u64 = 7039;

        let predicted =
            WEIGHTS_MIB + gemma4_12b().cache_mib(CacheType::Q8_0, 262_144) + COMPUTE_BUFFER_MIB;

        assert!(
            predicted >= MEASURED_MIB,
            "predicted {predicted} MiB but the card actually used {MEASURED_MIB} MiB - \
             under-estimating is what crashes the desktop"
        );
        assert!(
            predicted <= MEASURED_MIB * 11 / 10,
            "predicted {predicted} MiB against {MEASURED_MIB} measured - so conservative \
             it would refuse loads that work"
        );
    }

    #[test]
    fn weights_that_do_not_fit_yield_no_context() {
        let budget = Budget::new(900);
        // devstral-24b Q4_K_M, 13.34 GiB. Measured: could not hold even 4096.
        assert_eq!(
            budget.max_context(&qwen36_27b(), CacheType::Q8_0, 13_660),
            None
        );
    }

    #[test]
    fn a_desktop_over_the_ceiling_leaves_nothing() {
        assert_eq!(Budget::new(CEILING_MIB + 1).available_mib(), 0);
    }

    fn device(backend: crate::device::Backend, total: u64, free: u64) -> crate::device::Device {
        crate::device::Device {
            id: "D0".into(),
            name: "test".into(),
            backend,
            total_mib: total,
            free_mib: free,
        }
    }

    /// A discrete card must hold back enough for the compositor to keep allocating.
    #[test]
    fn a_discrete_device_reserves_room_for_the_desktop() {
        use crate::device::Backend;
        let b = Budget::for_device(&device(Backend::Vulkan, 16_384, 15_400));
        assert_eq!(b.ceiling_mib, 16_384 - DISCRETE_RESERVE_MIB);
        // 984 MiB already resident, so that is the baseline.
        assert_eq!(b.desktop_mib, 984);
        // Close to the hand-tuned ceiling this project was built against.
        assert!((b.ceiling_mib as i64 - CEILING_MIB as i64).abs() < 200);
    }

    /// Unified memory holds back a share of the machine, not a token amount.
    ///
    /// The reserve here is everything the operating system, the browser and the
    /// editor get. A flat 1024 MiB read as generous against a 16 GB card and left a
    /// 64 GB Mac 17 GB into swap, which is the failure this scales to avoid.
    #[test]
    fn a_unified_device_reserves_a_share_of_its_total() {
        use crate::device::Backend;
        let metal = Budget::for_device(&device(Backend::Metal, 26_000, 25_900));
        assert_eq!(metal.ceiling_mib, 26_000 - 6_500);

        // Still not a cap that stops a big Mac being a big Mac: three quarters of a
        // 52 GB working set is far more than any discrete card here offers.
        let big = Budget::for_device(&device(Backend::Metal, 53_084, 53_000));
        assert!(big.available_mib() > 38_000, "{}", big.available_mib());
    }

    /// A cap only ever lowers the answer, and never raises it past what fits.
    #[test]
    fn a_context_cap_applies_on_top_of_what_memory_allows() {
        use crate::device::Backend;
        let mac = Budget::for_device(&device(Backend::Metal, 53_084, 53_000));
        let kv = KvLayout::dense(48, 4, 128, 128);
        let uncapped = mac.max_context(&kv, CacheType::Q8_0, 17_697).unwrap();
        assert!(uncapped > 131_072, "{uncapped}");

        let capped = mac.capped_at(Some(131_072));
        assert_eq!(
            capped.max_context(&kv, CacheType::Q8_0, 17_697),
            Some(131_072)
        );

        // A cap above what fits changes nothing - memory still decides.
        let generous = mac.capped_at(Some(1 << 30));
        assert_eq!(
            generous.max_context(&kv, CacheType::Q8_0, 17_697),
            Some(uncapped)
        );
    }

    /// A 36 GB Mac should be able to run things this 16 GB card cannot.
    #[test]
    fn a_large_unified_device_affords_a_much_bigger_model() {
        use crate::device::Backend;
        let mac = Budget::for_device(&device(Backend::Metal, 36_864, 36_000));
        let kv = KvLayout::dense(65, 4, 256, 256);
        // qwen3.6-27b at Q4_K_M, which cannot load at all on 16 GB.
        assert!(mac.max_context(&kv, CacheType::Q8_0, 16_058).is_some());
    }

    #[test]
    fn cache_types_round_trip_through_their_llama_spelling() {
        for c in [CacheType::F16, CacheType::Q8_0, CacheType::Q4_0] {
            assert_eq!(c.as_llama_arg().parse::<CacheType>().unwrap(), c);
        }
        assert!("q3_k_m".parse::<CacheType>().is_err());
    }
}
