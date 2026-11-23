use slugkit::slugify;

#[test]
fn joins_words_with_dashes() {
	assert_eq!(slugify("Hello World"), "hello-world");
}

#[test]
fn drops_leading_punctuation() {
	assert_eq!(slugify("  --Hello"), "hello");
}

#[test]
fn collapses_runs_of_separators() {
	assert_eq!(slugify("a  -- b"), "a-b");
}

#[test]
fn keeps_digits() {
	assert_eq!(slugify("Top 10 Tips"), "top-10-tips");
}

#[test]
fn apostrophes_split_words() {
	assert_eq!(slugify("rock'n'roll"), "rock-n-roll");
}
