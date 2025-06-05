# AGENTS Instructions

These guidelines help maintain code quality and readability across the repository.

## Commit Messages
- Follow the Conventional Commits style: `<type>(<scope>): <subject>`.
- Common types include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, and `chore`.
- Keep the summary under 50 characters and write it in the imperative mood.
- Add a blank line before any body content and wrap body lines at 72 characters.
- Reference relevant issues in the body, e.g. `Fixes #123`.

## Testing Instructions
- Format code with `black` and run a linter (`ruff` or `flake8`) for code changes.
- Execute `pytest -q` to ensure all tests pass.
- Skip these steps only when modifying comments or documentation.

## Code Style
- Use `black`'s 88-character line length and 4-space indentation.
- Prefer type hints for new or updated functions and variables.
- Include docstrings for public functions and classes.

## PR Expectations
- Clearly summarize what changed and why in the PR description.
- Paste relevant test output in the PR.
- Link any related issues.
- Note deviations from these instructions if necessary.
