## What this change does

<!-- One or two sentences describing what changed and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature (please open an issue first to confirm scope)
- [ ] Documentation
- [ ] Refactor / cleanup (no behavior change)
- [ ] Build / tooling

## How I tested it

<!-- Commands you ran, manual steps you performed. -->

- [ ] `python -m pytest tests/` passes locally
- [ ] App still starts: `python -c "from src.app import App; app=App(); app.update(); print('STARTUP_SMOKE_OK'); app.destroy()"`
- [ ] If a UI page was touched, that page still loads (`_switch_page('<name>')`)

## Checklist

- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] I read and agree to the [Code of Conduct](../CODE_OF_CONDUCT.md)
- [ ] I understand this is a hobby project and the maintainer may take time to respond
- [ ] This change does not add telemetry, analytics, or "phone home" behavior
- [ ] This change does not require shipping model weights with the repo
- [ ] This change does not weaken the language in [DISCLAIMER.md](../DISCLAIMER.md)
