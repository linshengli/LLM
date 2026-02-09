# Diff Patch

Apply a unified diff or patch description to the codebase.

## Arguments

- `$ARGUMENTS` — Either a file path containing the diff/patch, or an inline description of changes to apply

## Instructions

1. Parse `$ARGUMENTS` to determine the input mode:
   - **File path**: Read the diff/patch file (supports unified diff, git diff, or `.patch` format)
   - **Inline description**: Parse the natural language description of changes

2. For unified diff / patch file input:
   - Parse each hunk: identify target file, line ranges, additions (+), deletions (-)
   - Validate that context lines match the current file content
   - If context doesn't match, report the mismatch and attempt fuzzy matching (offset tolerance ±5 lines)
   - Apply each hunk sequentially using the Edit tool

3. For inline description input:
   - Identify target files and the changes described
   - Read each target file
   - Apply the described modifications using Edit tool

4. After applying all changes:
   - Run `git diff` to show what was actually modified
   - If any hunk failed to apply, report it clearly with the conflicting context
   - Verify no syntax errors were introduced (run `python -c "import ast; ast.parse(open('<file>').read())"` for Python files)

5. Print summary: number of hunks applied, files modified, any failures.

## Supported Formats

- Unified diff (`diff -u` / `git diff` output)
- Git-format patches (`git format-patch` output)
- Natural language change descriptions
- Cherry-picked changes from specific commits (provide commit hash)
