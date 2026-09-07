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
}

/// The attention geometry that determines KV cache cost, read from GGUF metadata.
#[derive(Debug, Clone, Copy)]
pub struct KvLayout {
    pub layers: u32,
    pub kv_heads: u32,
    pub key_length: u32,
    pub value_length: u32,
}

impl KvLayout {
    /// Bytes of KV cache per token of context.
    ///
    /// Both K and V are stored for every layer and every KV head, so this is linear in
    /// all four dimensions. Grouped-query attention shows up as `kv_heads` being well
    /// below the model's attention head count.
    ///
    /// This *over*-estimates sliding-window models such as gemma4, where most layers
    /// hold a fixed-size window rather than growing with context. Over-estimating is
    /// the safe direction: it refuses loads that would in fact have fit.
    #[must_use]
    pub fn bytes_per_token(&self, cache: CacheType) -> f64 {
        let elements = f64::from(self.layers)
            * f64::from(self.kv_heads)
            * f64::from(self.key_length + self.value_length);
        elements * cache.bytes_per_element()
    }

    /// Total KV cache for `context` tokens, in MiB, rounded up.
    #[must_use]
    pub fn cache_mib(&self, cache: CacheType, context: u64) -> u64 {
        let bytes = self.bytes_per_token(cache) * context as f64;
        (bytes / (1024.0 * 1024.0)).ceil() as u64
    }
}

/// What a load has to fit inside.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// VRAM already held by the desktop, sampled before the load.
    pub desktop_mib: u64,
}

impl Budget {
    #[must_use]
    pub fn new(desktop_mib: u64) -> Self {
        Self { desktop_mib }
    }

    /// MiB available to llama-server for weights, KV cache and compute buffers.
    ///
    /// Saturates at zero rather than underflowing when the desktop alone is over the
    /// ceiling, which would mean nothing can be loaded at all.
    #[must_use]
    pub fn available_mib(&self) -> u64 {
        CEILING_MIB
            .saturating_sub(self.desktop_mib)
            .saturating_sub(COMPUTE_BUFFER_MIB)
    }

    /// Largest context that fits alongside `weights_mib`, or `None` if the weights
    /// alone do not fit.
    #[must_use]
    pub fn max_context(&self, kv: &KvLayout, cache: CacheType, weights_mib: u64) -> Option<u64> {
        let for_cache = self.available_mib().checked_sub(weights_mib)?;
        let per_token = kv.bytes_per_token(cache);
        if per_token <= 0.0 {
            return None;
        }
        Some((for_cache as f64 * 1024.0 * 1024.0 / per_token) as u64)
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
    const QWEN3_14B: KvLayout = KvLayout {
        layers: 40,
        kv_heads: 8,
        key_length: 128,
        value_length: 128,
    };

    /// qwen3.6-27b (arch `qwen35`): 65 layers, 4 KV heads, 256/256, global attention on
    /// every layer. Read from the first 8 MiB of the blob.
    const QWEN36_27B: KvLayout = KvLayout {
        layers: 65,
        kv_heads: 4,
        key_length: 256,
        value_length: 256,
    };

    #[test]
    fn kv_per_token_matches_the_measured_card() {
        let kib = QWEN3_14B.bytes_per_token(CacheType::Q8_0) / 1024.0;
        assert!(
            (kib - 84.8).abs() < 1.0,
            "predicted {kib:.1} KiB/token, measured 84.8"
        );
    }

    #[test]
    fn q8_0_cache_is_about_half_of_f16() {
        let f16 = QWEN3_14B.bytes_per_token(CacheType::F16);
        let q8 = QWEN3_14B.bytes_per_token(CacheType::Q8_0);
        assert!((f16 / q8 - 1.882).abs() < 0.01);
    }

    /// The load that actually killed the desktop: qwen3-14b at ctx=65536.
    #[test]
    fn rejects_the_load_that_crashed_the_machine() {
        let budget = Budget::new(900);
        assert!(
            !budget.fits(&QWEN3_14B, CacheType::Q8_0, 8836, 65536),
            "must refuse the ctx=65536 load that took the compositor down"
        );
        assert!(budget.fits(&QWEN3_14B, CacheType::Q8_0, 8836, 32768));
    }

    /// A 27B cannot hold a long context here at any quant, because quantising weights
    /// does not shrink the cache.
    #[test]
    fn dense_27b_cannot_reach_long_context() {
        let budget = Budget::new(900);
        let ctx = budget
            .max_context(&QWEN36_27B, CacheType::Q8_0, 10428)
            .expect("IQ3_XXS weights fit");
        assert!((18_000..21_000).contains(&ctx), "expected ~19k, got {ctx}");
        assert!(!budget.fits(&QWEN36_27B, CacheType::Q8_0, 10428, 200_000));
    }

    #[test]
    fn weights_that_do_not_fit_yield_no_context() {
        let budget = Budget::new(900);
        // devstral-24b Q4_K_M, 13.34 GiB. Measured: could not hold even 4096.
        assert_eq!(
            budget.max_context(&QWEN36_27B, CacheType::Q8_0, 13_660),
            None
        );
    }

    #[test]
    fn a_desktop_over_the_ceiling_leaves_nothing() {
        assert_eq!(Budget::new(CEILING_MIB + 1).available_mib(), 0);
    }
}
