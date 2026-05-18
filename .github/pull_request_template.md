## What does this PR do?

<!-- Describe the motivation and the changes made. Link to an issue if applicable. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring / cleanup
- [ ] CI/CD / tooling
- [ ] Documentation

## Testing

- [ ] Tested locally (`docker-compose up --build`)
- [ ] Webcam / mic pipeline tested end-to-end
- [ ] No automated tests needed for this change

## Security checklist

- [ ] No secrets, API keys, or credentials committed
- [ ] No new dependencies with known CVEs (`pip-audit` / `npm audit` checked)
- [ ] ANTHROPIC_API_KEY is read from environment, never hardcoded
- [ ] No SQL / command injection vectors introduced

## Architecture constraints (CLAUDE.md)

- [ ] Video stream stays at ≤ 1 FPS
- [ ] Claude API called exactly once per session (at Stop)
- [ ] No database / persistence added
- [ ] MediaPipe runs on backend only
- [ ] `anthropic==0.39.0` — no `output_config` used

## PR hygiene

- [ ] Commits are squashed / well-described
- [ ] No debug logs or `print()` statements left in
- [ ] Docker images still build (`Dockerfile.backend` / `Dockerfile.frontend`)
