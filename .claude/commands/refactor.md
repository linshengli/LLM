# Refactor

Systematically refactor code in the specified file or module.

## Arguments

- `$ARGUMENTS` — Target file/module path and refactoring goal (e.g., "src/models/transformer.py extract attention into separate class")

## Instructions

1. Parse the target path and refactoring intent from `$ARGUMENTS`.
2. Read the target file(s) completely. Understand the existing structure, dependencies, and call sites.
3. Before making any changes, produce a **Refactoring Plan**:
   - List each change as a numbered step
   - For each step, note: what moves/renames/extracts, which files are affected, potential breakage risks
   - Identify all import sites and call sites that need updating
4. Execute the refactoring plan step by step:
   - Use Edit tool for surgical changes (prefer over full rewrites)
   - Maintain all existing functionality (no behavior changes)
   - Update all imports, references, and type annotations
   - Preserve docstrings and comments; update them if signatures change
5. After all edits, verify:
   - No orphaned imports or undefined references
   - All `__init__.py` exports are consistent
   - Type hints remain valid
6. If tests exist, run them via `python -m pytest <test_path> -x --tb=short` and report results.
7. Print a summary: files modified, lines added/removed, key changes.

## Refactoring Types Supported

- **Extract**: Pull code into a new function/class/module
- **Inline**: Collapse unnecessary abstractions
- **Rename**: Rename symbols across the codebase consistently
- **Move**: Relocate functions/classes between modules
- **Simplify**: Reduce complexity while preserving semantics
- **Dedup**: Merge duplicated code into shared utilities
