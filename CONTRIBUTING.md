# Contributing to ASTRAUTOMA

Thanks for wanting to help! ASTRAUTOMA is a personal project, so the easiest way to help is a good bug report.

## Reporting a failed flight

Open an [issue](https://github.com/himsson/ASTRAUTOMA/issues/new/choose) and attach:

1. A screenshot of the flight table at the moment it went wrong.
2. The newest file from the `logs/` folder: every flight is logged in full.
3. Your KSP version, the list of mods, and the target you chose.
4. If you can, the `.craft` file of the rocket.

## Ideas

Use the *Feature request* template. Tell us what you want to fly and what stops you now.

## Code

1. Fork the repository and create a branch.
2. Keep every user-facing string bilingual: `L("English", "Русский")` from `astra/i18n.py`.
3. `ASTRAUTOMA.bat` must stay ASCII-only with CRLF line endings, because cmd.exe breaks otherwise.
4. Never commit weights (`*.pth`), your `config.json` or logs.
5. Test on a real game when you can, and say in the pull request what you flew.

By contributing, you agree that your work is published under the project's [license](LICENSE) (CC BY-NC 4.0).
