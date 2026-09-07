//! Manage local LLMs and expose them to coding harnesses.
//!
//! The binary is a thin shell over this library, so the logic that decides whether a
//! model may be loaded is testable without a GPU present.

pub mod vram;
