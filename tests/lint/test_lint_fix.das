options gen2

require dastest/testing_boost
require daslib/lint_config

[test]
def test_lint_fix(t : T?) {
    t |> run("overlapping and out-of-text spans are dropped") @@(t : T?) {
        let fixes <- [
            LintFix(line = 1, column = 0, last_line = 1, last_column = 3, replacement = "X"),
            LintFix(line = 1, column = 2, last_line = 1, last_column = 4, replacement = "Y"),
            LintFix(line = 9, column = 0, last_line = 9, last_column = 1, replacement = "Z")]
        let fixed = apply_lint_fixes("abcdef\n", fixes)
        t |> equal(fixed.applied, 1)
        t |> equal(fixed.text, "Xdef\n")
    }
}
