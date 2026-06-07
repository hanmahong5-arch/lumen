//! `lumen tour` — assemble a narrated index over per-capability lifecycle
//! exports.
//!
//! The capability tour (`kova-rest/scripts/capability-tour.sh`) drives each Kova
//! capability as a real run and exports one self-contained lifecycle HTML per
//! capability (`<label>.html`). This subcommand stitches those into one
//! shareable `index.html`: a card grid where each card narrates a capability and
//! links to its lifecycle file in the same directory. Pure static HTML/CSS
//! (reuses the export's visual theme) — no server, no CDN, opens offline.

/// The card-grid index shell. `__COUNT__` and `__CARDS__` are filled at render
/// time; a drift-guard test asserts both placeholders exist.
const TOUR_HTML: &str = include_str!("tour.html");

/// One capability card: a driven run, its label, and a one-line narration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TourEntry {
    /// Run id that was exported (shown as card metadata).
    pub run_id: String,
    /// Capability label — also the per-capability HTML filename stem.
    pub label: String,
    /// One-line narration of what this capability demonstrates.
    pub blurb: String,
}

/// Parse one `--entry run_id:label:blurb` value.
///
/// Splits on the first two colons only, so the blurb may itself contain colons
/// (e.g. a URL or `key: value` note). Requires at least `run_id` and `label`;
/// the blurb is optional (an empty trailing field is allowed). Surrounding
/// whitespace on `run_id`/`label` is trimmed. Returns `None` if `run_id` or
/// `label` is empty after trimming.
pub fn parse_entry(raw: &str) -> Option<TourEntry> {
    let mut parts = raw.splitn(3, ':');
    let run_id = parts.next()?.trim().to_string();
    let label = parts.next()?.trim().to_string();
    let blurb = parts.next().unwrap_or("").trim().to_string();
    if run_id.is_empty() || label.is_empty() {
        return None;
    }
    Some(TourEntry {
        run_id,
        label,
        blurb,
    })
}

/// Render the narrated tour index for `entries` into a self-contained HTML page.
pub fn render_tour(entries: &[TourEntry]) -> String {
    let cards = if entries.is_empty() {
        "    <div class=\"empty\">No capabilities exported.</div>".to_string()
    } else {
        entries.iter().map(card_html).collect::<Vec<_>>().join("\n")
    };
    // Fill `__COUNT__` before `__CARDS__` so an (escaped) card value that happens
    // to contain `__COUNT__` is never re-substituted. The cards string is the
    // replacement value, not the template, so it is never re-scanned.
    TOUR_HTML
        .replace("__COUNT__", &entries.len().to_string())
        .replace("__CARDS__", &cards)
}

/// One `<a class="tcard">` linking to `<label>.html` in the same directory.
fn card_html(e: &TourEntry) -> String {
    let href = format!("{}.html", sanitize_filename(&e.label));
    let blurb = if e.blurb.is_empty() {
        String::new()
    } else {
        format!("\n      <div class=\"blurb\">{}</div>", esc(&e.blurb))
    };
    format!(
        "    <a class=\"tcard\" href=\"{href}\">\n      \
         <div class=\"cap\">{label}</div>{blurb}\n      \
         <div class=\"meta\"><span>run {run}</span><span class=\"open\">open &rarr;</span></div>\n    </a>",
        href = esc_attr(&href),
        label = esc(&e.label),
        run = esc(&e.run_id),
    )
}

/// Reduce a label to a filesystem/href-safe stem: keep `[A-Za-z0-9_.-]`, map
/// anything else to `_`. The capability labels come from our own tour script
/// (`sleep`, `continue_as_new`, `tool_policy`, …) and so pass through verbatim;
/// this only hardens against an unexpected entry.
fn sanitize_filename(label: &str) -> String {
    let s: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "capability".to_string()
    } else {
        s
    }
}

/// Minimal HTML-text escaper (element content).
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// HTML attribute-value escaper (adds quote escaping on top of [`esc`]).
fn esc_attr(s: &str) -> String {
    esc(s).replace('"', "&quot;")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_entry_splits_three_fields() {
        let e = parse_entry("4815:sleep:durable timer wakes the run").unwrap();
        assert_eq!(e.run_id, "4815");
        assert_eq!(e.label, "sleep");
        assert_eq!(e.blurb, "durable timer wakes the run");
    }

    #[test]
    fn parse_entry_blurb_may_contain_colons() {
        // splitn(3) keeps everything after the second colon in the blurb.
        let e = parse_entry("42:http_call:GET http://x/y: bounded 1 MiB").unwrap();
        assert_eq!(e.run_id, "42");
        assert_eq!(e.label, "http_call");
        assert_eq!(e.blurb, "GET http://x/y: bounded 1 MiB");
    }

    #[test]
    fn parse_entry_blurb_optional() {
        let e = parse_entry("7:cancel").unwrap();
        assert_eq!(e.label, "cancel");
        assert_eq!(e.blurb, "");
    }

    #[test]
    fn parse_entry_rejects_missing_label_or_id() {
        assert!(parse_entry("loneword").is_none());
        assert!(parse_entry(":label:blurb").is_none(), "empty run_id");
        assert!(parse_entry("rid::blurb").is_none(), "empty label");
    }

    #[test]
    fn render_tour_links_each_capability_html() {
        let entries = vec![
            parse_entry("100:sleep:durable timer").unwrap(),
            parse_entry("200:continue_as_new:CAN chains a successor").unwrap(),
        ];
        let html = render_tour(&entries);
        assert!(html.contains("href=\"sleep.html\""));
        assert!(html.contains("href=\"continue_as_new.html\""));
        assert!(html.contains("durable timer"));
        assert!(html.contains("CAN chains a successor"));
        assert!(html.contains("run 100"));
        assert!(html.contains("2 capabilities"), "count is filled");
        // No unrendered placeholders survive.
        assert!(!html.contains("__CARDS__"));
        assert!(!html.contains("__COUNT__"));
    }

    #[test]
    fn render_tour_escapes_html_in_entries() {
        // A blurb/label with HTML-special chars must not break out into markup.
        let e = parse_entry("9:lab<el>:<script>alert(1)</script>").unwrap();
        let html = render_tour(&[e]);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
        // The href is sanitized (no raw `<`/`>` in the filename).
        assert!(html.contains("href=\"lab_el_.html\""));
    }

    #[test]
    fn render_tour_empty_is_clean() {
        let html = render_tour(&[]);
        assert!(html.contains("No capabilities exported."));
        assert!(html.contains("0 capabilities"));
        assert!(!html.contains("__CARDS__"));
    }

    #[test]
    fn sanitize_filename_keeps_safe_chars() {
        assert_eq!(sanitize_filename("crash_recovery"), "crash_recovery");
        assert_eq!(sanitize_filename("tool.policy-1"), "tool.policy-1");
        assert_eq!(sanitize_filename("a/b c"), "a_b_c");
        assert_eq!(sanitize_filename(""), "capability");
    }

    /// Drift guard: the shell must keep both fill points.
    #[test]
    fn tour_html_has_both_placeholders() {
        assert!(TOUR_HTML.contains("__COUNT__"), "missing __COUNT__");
        assert!(TOUR_HTML.contains("__CARDS__"), "missing __CARDS__");
    }
}
