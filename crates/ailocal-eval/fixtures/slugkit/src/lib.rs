//! Turn titles into URL slugs.

/// Lowercase `title`, keep its ASCII letters and digits, and join the words
/// with dashes.
pub fn slugify(title: &str) -> String {
	let mut slug = String::new();
	for c in title.chars() {
		if c.is_ascii_alphanumeric() {
			slug.push(c.to_ascii_lowercase());
		} else if !slug.is_empty() && !slug.ends_with('-') {
			slug.push('-');
		}
	}
	slug
}
