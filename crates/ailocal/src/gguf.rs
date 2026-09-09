//! Minimal GGUF metadata reader.
//!
//! Deliberately parses from a byte slice and treats running out of input as success
//! with partial results, rather than an error. GGUF puts all metadata at the head of
//! the file, so a ranged HTTP request for the first few MiB is enough to learn a
//! model's architecture - which lets us refuse a 28 GB download that could never fit
//! in VRAM. General-purpose GGUF crates mmap the whole file and cannot do that.
//!
//! Format: magic `GGUF`, u32 version, u64 tensor count, u64 metadata count, then that
//! many (key, type, value) triples. All little-endian.

use std::collections::BTreeMap;

use crate::vram::{KvLayout, LayerGroup};

/// Metadata value types, in GGUF's own numbering. The discriminants are the wire
/// format, so they must not be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl Kind {
    fn from_wire(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// The value as an unsigned integer, if it is one that fits.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Int(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Largest integer in this value, treating a scalar as a one-element array.
    ///
    /// Some architectures publish per-layer arrays where others publish a scalar -
    /// gemma4 gives `attention.head_count_kv` as 48 per-layer entries. Taking the
    /// maximum keeps the KV estimate on the safe (over-estimating) side.
    #[must_use]
    pub fn max_u64(&self) -> Option<u64> {
        match self {
            Self::Int(_) => self.as_u64(),
            Self::Array(items) => items.iter().filter_map(Value::as_u64).max(),
            _ => None,
        }
    }

    /// Per-element booleans, if this is an array of them.
    #[must_use]
    pub fn bool_vec(&self) -> Option<Vec<bool>> {
        match self {
            Self::Array(items) => items
                .iter()
                .map(|v| match v {
                    Self::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    /// Largest integer among the entries selected by `keep`.
    ///
    /// A scalar applies to every layer, so it is returned as-is.
    fn max_u64_where(&self, keep: impl Fn(usize) -> bool) -> Option<u64> {
        match self {
            Self::Int(_) => self.as_u64(),
            Self::Array(items) => items
                .iter()
                .enumerate()
                .filter(|(i, _)| keep(*i))
                .filter_map(|(_, v)| v.as_u64())
                .max(),
            _ => None,
        }
    }
}

/// Everything we could read out of a GGUF head.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub version: u32,
    pub tensor_count: u64,
    /// True when the input ran out before all declared keys were read.
    pub truncated: bool,
    keys: BTreeMap<String, Value>,
}

impl Metadata {
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.keys.get(key)
    }

    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture")?.as_str()
    }

    /// Look up a key under the model's own architecture prefix.
    ///
    /// GGUF namespaces per-architecture keys by the value of `general.architecture`,
    /// which is not always the family name - qwen3.6-27b reports `qwen35`.
    #[must_use]
    pub fn arch_key(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }

    /// The context length the model was trained for.
    #[must_use]
    pub fn context_length(&self) -> Option<u64> {
        self.arch_key("context_length")?.as_u64()
    }

    /// Attention geometry needed to size the KV cache.
    ///
    /// Falls back to `embedding_length / head_count` for key and value length, which is
    /// how architectures that omit the explicit fields define head dimension.
    ///
    /// When the model publishes a sliding-window pattern, the layers are split into the
    /// global ones that grow with context and the windowed ones that do not. gemma4
    /// makes this worth the trouble: 40 of its 48 blocks are windowed, and its 8 global
    /// blocks carry a single KV head against the windowed layers' eight.
    #[must_use]
    pub fn kv_layout(&self) -> Option<KvLayout> {
        let layers = u32::try_from(self.arch_key("block_count")?.max_u64()?).ok()?;
        let heads = self.arch_key("attention.head_count_kv")?;

        let head_dim = || -> Option<u64> {
            let embd = self.arch_key("embedding_length")?.max_u64()?;
            let n = self.arch_key("attention.head_count")?.max_u64()?;
            embd.checked_div(n)
        };
        let dim = |key: &str| -> Option<u32> {
            let v = self
                .arch_key(key)
                .and_then(Value::max_u64)
                .or_else(head_dim)?;
            u32::try_from(v).ok()
        };
        let key_length = dim("attention.key_length")?;
        let value_length = dim("attention.value_length")?;

        let pattern = self
            .arch_key("attention.sliding_window_pattern")
            .and_then(Value::bool_vec);
        let window = self
            .arch_key("attention.sliding_window")
            .and_then(Value::max_u64)
            .and_then(|w| u32::try_from(w).ok());

        // Without a per-layer pattern we cannot tell which layers are windowed, so
        // treat every layer as global. That over-estimates, which is the safe way to
        // be wrong.
        let (Some(pattern), Some(window)) = (pattern, window) else {
            let kv_heads = u32::try_from(heads.max_u64()?).ok()?;
            return Some(KvLayout::dense(layers, kv_heads, key_length, value_length));
        };
        if window == 0 || pattern.iter().all(|swa| !swa) {
            let kv_heads = u32::try_from(heads.max_u64()?).ok()?;
            return Some(KvLayout::dense(layers, kv_heads, key_length, value_length));
        }

        let n_sliding = u32::try_from(pattern.iter().filter(|swa| **swa).count()).ok()?;
        let n_global = layers.saturating_sub(n_sliding);

        let global_heads =
            u32::try_from(heads.max_u64_where(|i| !pattern.get(i).copied().unwrap_or(false))?)
                .ok()?;
        let sliding_heads =
            u32::try_from(heads.max_u64_where(|i| pattern.get(i).copied().unwrap_or(false))?)
                .ok()?;

        let swa_key = dim("attention.key_length_swa").unwrap_or(key_length);
        let swa_value = dim("attention.value_length_swa").unwrap_or(value_length);

        Some(KvLayout {
            global: LayerGroup {
                layers: n_global,
                kv_heads: global_heads,
                key_length,
                value_length,
            },
            sliding: Some((
                LayerGroup {
                    layers: n_sliding,
                    kv_heads: sliding_heads,
                    key_length: swa_key,
                    value_length: swa_value,
                },
                window,
            )),
        })
    }

    /// Whether the model uses sliding-window attention, which decouples most of its KV
    /// cache from context length.
    #[must_use]
    pub fn has_sliding_window(&self) -> bool {
        self.arch_key("attention.sliding_window")
            .and_then(Value::max_u64)
            .is_some_and(|w| w > 0)
    }
}

/// A cursor that reports exhaustion rather than panicking, so a truncated head is a
/// normal outcome instead of an error.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<String> {
        let len = usize::try_from(self.u64()?).ok()?;
        Some(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn value(&mut self, kind: Kind) -> Option<Value> {
        Some(match kind {
            Kind::U8 => Value::Int(i128::from(self.take(1)?[0])),
            Kind::I8 => Value::Int(i128::from(self.take(1)?[0] as i8)),
            Kind::U16 => Value::Int(i128::from(u16::from_le_bytes(
                self.take(2)?.try_into().ok()?,
            ))),
            Kind::I16 => Value::Int(i128::from(i16::from_le_bytes(
                self.take(2)?.try_into().ok()?,
            ))),
            Kind::U32 => Value::Int(i128::from(self.u32()?)),
            Kind::I32 => Value::Int(i128::from(i32::from_le_bytes(
                self.take(4)?.try_into().ok()?,
            ))),
            Kind::F32 => Value::Float(f64::from(f32::from_le_bytes(
                self.take(4)?.try_into().ok()?,
            ))),
            Kind::Bool => Value::Bool(self.take(1)?[0] != 0),
            Kind::String => Value::String(self.string()?),
            Kind::U64 => Value::Int(i128::from(self.u64()?)),
            Kind::I64 => Value::Int(i128::from(i64::from_le_bytes(
                self.take(8)?.try_into().ok()?,
            ))),
            Kind::F64 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().ok()?)),
            Kind::Array => {
                let elem = Kind::from_wire(self.u32()?)?;
                let len = usize::try_from(self.u64()?).ok()?;
                // Tokenizer vocabularies are arrays of ~150k strings. Walk them to stay
                // aligned, but do not retain them.
                let keep = len <= 4096;
                let mut items = Vec::with_capacity(if keep { len } else { 0 });
                for _ in 0..len {
                    let v = self.value(elem)?;
                    if keep {
                        items.push(v);
                    }
                }
                Value::Array(items)
            }
        })
    }
}

/// Parse metadata from the head of a GGUF file.
///
/// Returns `Ok` with `truncated: true` when `bytes` ends mid-metadata, since that is
/// the expected case for a ranged download.
///
/// # Errors
/// If the input is not GGUF, or is too short to hold even the header.
pub fn parse_head(bytes: &[u8]) -> anyhow::Result<Metadata> {
    let mut c = Cursor { bytes, pos: 0 };

    let magic = c
        .take(4)
        .ok_or_else(|| anyhow::anyhow!("input too short"))?;
    anyhow::ensure!(magic == b"GGUF", "not a GGUF file (magic was {magic:?})");

    let version = c.u32().ok_or_else(|| anyhow::anyhow!("truncated header"))?;
    let tensor_count = c.u64().ok_or_else(|| anyhow::anyhow!("truncated header"))?;
    let count = c.u64().ok_or_else(|| anyhow::anyhow!("truncated header"))?;

    let mut keys = BTreeMap::new();
    let mut truncated = false;
    for _ in 0..count {
        let Some(entry) = (|| {
            let key = c.string()?;
            let kind = Kind::from_wire(c.u32()?)?;
            Some((key, c.value(kind)?))
        })() else {
            truncated = true;
            break;
        };
        keys.insert(entry.0, entry.1);
    }

    Ok(Metadata {
        version,
        tensor_count,
        truncated,
        keys,
    })
}

/// Read metadata from the first `limit` bytes of a file on disk.
///
/// # Errors
/// If the file cannot be read, or does not parse as GGUF.
/// The chat-template variable llama.cpp uses to turn thinking on and off per request.
const THINKING_TOGGLE: &[u8] = b"enable_thinking";

/// The one llama.cpp passes an effort *level* to, for templates that take one.
const THINKING_EFFORT: &[u8] = b"reasoning_effort";

/// How far in to look for them.
///
/// The chat template is metadata, so it is near the front of the file - but it sits
/// *after* the tokeniser vocabulary, which for a 12B is several megabytes. gemma4's
/// template starts past 8 MB, which is why the ordinary metadata read does not reach
/// it. 64 MB clears both models here with room to spare, and is a sequential read of a
/// file that is almost always in page cache.
const TEMPLATE_SEARCH_BYTES: usize = 64 << 20;

/// What per-request thinking control a model's chat template exposes.
///
/// This is the model's own capability, not a setting of ours, and a harness has to be
/// told which of the three it is: reporting a dial on a model that has only a switch
/// offers controls that provably do nothing, and reporting nothing on a model that can
/// think leaves the harness offering `off` as the only choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    /// No thinking mode at all. qwen3-coder-30b is one of these.
    None,
    /// A boolean switch, `chat_template_kwargs.enable_thinking`. gemma4-12b.
    Toggle,
    /// An effort level the template itself reads, via llama.cpp's `reasoning_effort`.
    Effort,
}

impl Thinking {
    /// Whether the model can think at all.
    #[must_use]
    pub const fn is_available(self) -> bool {
        !matches!(self, Self::None)
    }

    /// One word for a table or a status line.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "-",
            Self::Toggle => "on/off",
            Self::Effort => "effort",
        }
    }
}

/// Which thinking control this model's chat template exposes.
///
/// Deliberately a substring search rather than a full metadata parse. The template is
/// the last thing in the metadata and reaching it properly means parsing every
/// preceding key including the vocabulary, which is most of the cost of reading the
/// file. Both names are specific enough that finding one in the header is conclusive,
/// and this is asked once when writing a harness catalogue, not on the hot path.
///
/// It agrees with llama.cpp's own answer: `/props` reports `chat_template_caps`, and
/// for both models here `supports_reasoning_effort` matches what this returns.
#[must_use]
pub fn thinking_support(path: &std::path::Path) -> Thinking {
    use std::io::Read as _;

    let Ok(file) = std::fs::File::open(path) else {
        return Thinking::None;
    };
    let mut buf = Vec::new();
    if file
        .take(TEMPLATE_SEARCH_BYTES as u64)
        .read_to_end(&mut buf)
        .is_err()
    {
        return Thinking::None;
    }
    thinking_in_header(&buf)
}

/// The search itself, so it can be tested without a multi-gigabyte fixture.
fn thinking_in_header(bytes: &[u8]) -> Thinking {
    let holds = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);

    // Effort first: a template that takes a level can also be switched off, and the
    // level is the more capable of the two answers.
    if holds(THINKING_EFFORT) {
        Thinking::Effort
    } else if holds(THINKING_TOGGLE) {
        Thinking::Toggle
    } else {
        Thinking::None
    }
}

pub fn read_file(path: &std::path::Path, limit: usize) -> anyhow::Result<Metadata> {
    use std::io::Read as _;

    let mut buf = Vec::with_capacity(limit.min(8 << 20));
    std::fs::File::open(path)?
        .take(limit as u64)
        .read_to_end(&mut buf)?;
    parse_head(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a GGUF head by hand so the parser is tested without a 7 GB fixture.
    struct Builder {
        out: Vec<u8>,
        count: u64,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                count: 0,
            }
        }

        fn str_bytes(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }

        fn string(mut self, key: &str, val: &str) -> Self {
            Self::str_bytes(&mut self.out, key);
            self.out
                .extend_from_slice(&(Kind::String as u32).to_le_bytes());
            Self::str_bytes(&mut self.out, val);
            self.count += 1;
            self
        }

        fn u32(mut self, key: &str, val: u32) -> Self {
            Self::str_bytes(&mut self.out, key);
            self.out
                .extend_from_slice(&(Kind::U32 as u32).to_le_bytes());
            self.out.extend_from_slice(&val.to_le_bytes());
            self.count += 1;
            self
        }

        fn u32_array(mut self, key: &str, vals: &[u32]) -> Self {
            Self::str_bytes(&mut self.out, key);
            self.out
                .extend_from_slice(&(Kind::Array as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(Kind::U32 as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.out.extend_from_slice(&v.to_le_bytes());
            }
            self.count += 1;
            self
        }

        fn bool_array(mut self, key: &str, vals: &[bool]) -> Self {
            Self::str_bytes(&mut self.out, key);
            self.out
                .extend_from_slice(&(Kind::Array as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(Kind::Bool as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.out.push(u8::from(*v));
            }
            self.count += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut head = Vec::new();
            head.extend_from_slice(b"GGUF");
            head.extend_from_slice(&3u32.to_le_bytes());
            head.extend_from_slice(&866u64.to_le_bytes());
            head.extend_from_slice(&self.count.to_le_bytes());
            head.extend_from_slice(&self.out);
            head
        }
    }

    /// qwen3.6-27b as actually observed: architecture is `qwen35`, not `qwen3.6`.
    fn qwen36_27b() -> Vec<u8> {
        Builder::new()
            .string("general.architecture", "qwen35")
            .u32("qwen35.block_count", 65)
            .u32("qwen35.context_length", 262_144)
            .u32("qwen35.attention.head_count", 24)
            .u32("qwen35.attention.head_count_kv", 4)
            .u32("qwen35.attention.key_length", 256)
            .u32("qwen35.attention.value_length", 256)
            .build()
    }

    #[test]
    fn reads_header_fields() {
        let md = parse_head(&qwen36_27b()).unwrap();
        assert_eq!(md.version, 3);
        assert_eq!(md.tensor_count, 866);
        assert!(!md.truncated);
        assert_eq!(md.architecture(), Some("qwen35"));
        assert_eq!(md.context_length(), Some(262_144));
    }

    #[test]
    fn derives_the_kv_layout_that_rules_out_a_27b() {
        let md = parse_head(&qwen36_27b()).unwrap();
        let kv = md.kv_layout().unwrap();
        assert_eq!(kv.global.layers, 65);
        assert_eq!(kv.global.kv_heads, 4);
        assert!(
            kv.sliding.is_none(),
            "qwen35 attends globally on every layer"
        );

        let kib = kv.global.bytes_per_token(crate::vram::CacheType::Q8_0) / 1024.0;
        assert!(
            (kib - 138.0).abs() < 1.0,
            "expected ~138 KiB/token, got {kib:.1}"
        );
    }

    /// The real reason this parser exists: decide from a ranged request.
    #[test]
    fn truncation_is_partial_success_not_failure() {
        let full = qwen36_27b();
        let md = parse_head(&full[..full.len() - 12]).unwrap();
        assert!(md.truncated, "must flag the shortfall");
        assert_eq!(md.architecture(), Some("qwen35"));
        assert_eq!(md.context_length(), Some(262_144));
    }

    /// Without a per-layer pattern we cannot tell which blocks are windowed, so every
    /// block is treated as global and per-layer head counts collapse to their maximum.
    /// Over-estimating is the safe way to be wrong.
    #[test]
    fn per_layer_arrays_collapse_to_their_maximum() {
        let head = Builder::new()
            .string("general.architecture", "gemma4")
            .u32("gemma4.block_count", 48)
            .u32_array("gemma4.attention.head_count_kv", &[2, 4, 2, 8])
            .u32("gemma4.attention.key_length", 512)
            .u32("gemma4.attention.value_length", 512)
            .build();
        let kv = parse_head(&head).unwrap().kv_layout().unwrap();
        assert_eq!(kv.global.kv_heads, 8);
        assert!(kv.sliding.is_none());
    }

    /// gemma4-12b as it really is: 5 windowed blocks then 1 global, repeated 8 times,
    /// with 8 KV heads on the windowed blocks and 1 on the global ones. Getting this
    /// split wrong is a 40x error in its KV cost, which is what made the first version
    /// of `model ls` claim the daily driver could only hold 15k tokens.
    #[test]
    fn splits_gemma4_into_global_and_windowed_layers() {
        let pattern: Vec<bool> = (0..48).map(|i| (i + 1) % 6 != 0).collect();
        let heads: Vec<u32> = pattern.iter().map(|swa| if *swa { 8 } else { 1 }).collect();

        let head = Builder::new()
            .string("general.architecture", "gemma4")
            .u32("gemma4.block_count", 48)
            .u32_array("gemma4.attention.head_count_kv", &heads)
            .u32("gemma4.attention.key_length", 512)
            .u32("gemma4.attention.value_length", 512)
            .u32("gemma4.attention.key_length_swa", 256)
            .u32("gemma4.attention.value_length_swa", 256)
            .u32("gemma4.attention.sliding_window", 1024)
            .bool_array("gemma4.attention.sliding_window_pattern", &pattern)
            .build();

        let md = parse_head(&head).unwrap();
        assert!(md.has_sliding_window());
        let kv = md.kv_layout().unwrap();

        assert_eq!(kv.global.layers, 8);
        assert_eq!(
            kv.global.kv_heads, 1,
            "global blocks carry a single KV head"
        );
        assert_eq!(kv.global.key_length, 512);

        let (swa, window) = kv.sliding.expect("windowed layers");
        assert_eq!(window, 1024);
        assert_eq!(swa.layers, 40);
        assert_eq!(swa.kv_heads, 8);
        assert_eq!(swa.key_length, 256);

        // Must land near the measured marginal cost, not the 408 KiB/token that
        // treating every block as global produces.
        let kib = kv.global.bytes_per_token(crate::vram::CacheType::Q8_0) / 1024.0;
        assert!(
            (kib - 8.5).abs() < 1.0,
            "expected ~8.5 KiB/token, got {kib:.1}"
        );
    }

    #[test]
    fn head_dim_falls_back_to_embedding_over_heads() {
        let head = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.embedding_length", 4096)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .build();
        let kv = parse_head(&head).unwrap().kv_layout().unwrap();
        assert_eq!(kv.global.key_length, 128);
        assert_eq!(kv.global.value_length, 128);
    }

    #[test]
    fn rejects_non_gguf() {
        assert!(parse_head(b"NOPE____").is_err());
        assert!(parse_head(b"GG").is_err());
    }

    /// The measured shape of both models here: gemma4's template mentions
    /// `enable_thinking`, qwen3-coder's mentions neither, and llama-server's own
    /// `chat_template_caps` agrees with both answers.
    #[test]
    fn a_template_with_a_switch_reports_a_switch() {
        let head = Builder::new()
            .string(
                "tokenizer.chat_template",
                "{% if enable_thinking %}<think>{% endif %}",
            )
            .build();
        assert_eq!(thinking_in_header(&head), Thinking::Toggle);
    }

    #[test]
    fn a_template_with_no_thinking_reports_none() {
        let head = Builder::new()
            .string("tokenizer.chat_template", "{{ messages[0].content }}")
            .build();
        assert_eq!(thinking_in_header(&head), Thinking::None);
    }

    /// A level is the more capable answer, so a template offering both is a dial.
    #[test]
    fn an_effort_dial_outranks_the_switch_it_also_has() {
        let head = Builder::new()
            .string(
                "tokenizer.chat_template",
                "{% if enable_thinking %}{{ reasoning_effort }}{% endif %}",
            )
            .build();
        assert_eq!(thinking_in_header(&head), Thinking::Effort);
    }

    /// A file that cannot be read is not a model that can think - and must not be a
    /// panic either, since this runs over whatever is in the models directory.
    #[test]
    fn an_unreadable_file_reports_no_thinking() {
        assert_eq!(
            thinking_support(std::path::Path::new("/nonexistent/model.gguf")),
            Thinking::None
        );
    }

    #[test]
    fn labels_distinguish_the_three_answers() {
        assert_eq!(Thinking::None.label(), "-");
        assert!(!Thinking::None.is_available());
        assert!(Thinking::Toggle.is_available());
        assert!(Thinking::Effort.is_available());
    }
}
