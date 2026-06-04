//! LLM model pricing table and cost estimation.
//!
//! Provides per-token pricing for popular models and a fuzzy-match
//! estimator that works even when the exact model ID isn't in the table.

/// Pricing entry: (prompt_per_1k_usd, completion_per_1k_usd).
type PricingEntry = (f64, f64);

/// Default pricing for unknown models (conservative estimate).
const FALLBACK_PRICING: PricingEntry = (1.0, 3.0);

/// Static pricing table: (model_prefix, prompt_per_1k, completion_per_1k).
const PRICING_TABLE: &[(&str, f64, f64)] = &[
    // Anthropic Claude 4.x
    ("claude-opus-4-6", 15.0, 75.0),
    ("claude-sonnet-4-6", 3.0, 15.0),
    ("claude-haiku-4-5", 0.80, 4.0),
    // Anthropic Claude 3.x
    ("claude-3-5-sonnet", 3.0, 15.0),
    ("claude-3-5-haiku", 0.80, 4.0),
    ("claude-3-opus", 15.0, 75.0),
    ("claude-3-sonnet", 3.0, 15.0),
    ("claude-3-haiku", 0.25, 1.25),
    // OpenAI GPT-4o
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.0),
    // OpenAI GPT-4
    ("gpt-4-turbo", 10.0, 30.0),
    ("gpt-4", 30.0, 60.0),
    // OpenAI GPT-3.5
    ("gpt-3.5-turbo", 0.50, 1.50),
    // OpenAI o-series
    ("o1-mini", 3.0, 12.0),
    ("o1-preview", 15.0, 60.0),
    ("o1", 15.0, 60.0),
    ("o3-mini", 1.10, 4.40),
    ("o3", 10.0, 40.0),
    ("o4-mini", 1.10, 4.40),
    // Google Gemini
    ("gemini-2.0-flash", 0.10, 0.40),
    ("gemini-2.5-pro", 1.25, 10.0),
    ("gemini-1.5-pro", 1.25, 5.0),
    ("gemini-1.5-flash", 0.075, 0.30),
    // Meta Llama
    ("llama-3.3-70b", 0.88, 0.88),
    ("llama-3.1-70b", 0.88, 0.88),
    ("llama-3.1-8b", 0.18, 0.18),
    // Mistral
    ("mistral-large", 2.0, 6.0),
    ("mistral-small", 0.20, 0.60),
    // DeepSeek (deepseek-chat is the V3 chat endpoint — same pricing)
    ("deepseek-chat", 0.27, 1.10),
    ("deepseek-reasoner", 0.55, 2.19),
    ("deepseek-v3", 0.27, 1.10),
    ("deepseek-r1", 0.55, 2.19),
];

/// Estimate cost in USD for a given model and token counts.
///
/// Uses prefix matching: `"claude-sonnet-4-6-20260301"` matches `"claude-sonnet-4-6"`.
/// Falls back to $1.00/$3.00 per 1K tokens for unknown models.
#[must_use]
pub fn estimate_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let (prompt_price, completion_price) = lookup_pricing(model);
    (f64::from(prompt_tokens) * prompt_price + f64::from(completion_tokens) * completion_price)
        / 1000.0
}

/// Look up pricing with prefix fuzzy matching.
fn lookup_pricing(model: &str) -> PricingEntry {
    let model_lower = model.to_lowercase();

    // Prefix match: find the longest matching prefix
    let mut best: Option<(usize, PricingEntry)> = None;
    for &(prefix, prompt, completion) in PRICING_TABLE {
        if model_lower.starts_with(prefix) {
            let len = prefix.len();
            if best.is_none() || len > best.map_or(0, |(l, _)| l) {
                best = Some((len, (prompt, completion)));
            }
        }
    }

    if let Some((_, pricing)) = best {
        return pricing;
    }

    // Substring match: check if model contains a known prefix
    for &(prefix, prompt, completion) in PRICING_TABLE {
        if model_lower.contains(prefix) {
            return (prompt, completion);
        }
    }

    FALLBACK_PRICING
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let cost = estimate_cost("claude-sonnet-4-6", 1000, 100);
        let expected = (1000.0 * 3.0 + 100.0 * 15.0) / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn prefix_match() {
        let cost = estimate_cost("claude-sonnet-4-6-20260301", 1000, 0);
        let expected = 1000.0 * 3.0 / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn gpt4o_before_gpt4o_mini() {
        // "gpt-4o" should not match "gpt-4o-mini" entry
        let cost = estimate_cost("gpt-4o", 1000, 0);
        let expected = 1000.0 * 2.5 / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn gpt4o_mini_exact() {
        let cost = estimate_cost("gpt-4o-mini", 1000, 0);
        let expected = 1000.0 * 0.15 / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn unknown_model_fallback() {
        let cost = estimate_cost("totally-unknown-xyz", 1000, 500);
        let expected = (1000.0 * 1.0 + 500.0 * 3.0) / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn zero_tokens() {
        assert_eq!(estimate_cost("gpt-4o", 0, 0), 0.0);
    }

    #[test]
    fn substring_match() {
        let cost = estimate_cost("accounts/fireworks/models/llama-3.1-70b", 1000, 0);
        let expected = 1000.0 * 0.88 / 1000.0;
        assert!((cost - expected).abs() < 0.001);
    }
}
