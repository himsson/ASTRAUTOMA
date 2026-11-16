# Contributing

This is my own project so the most useful thing is a good bug report.

If a flight went wrong, open an issue and attach a screenshot of the flight table, the newest
file from `logs/`, your KSP version and mods, and the .craft if you can.

If you want to send code:

- every string the user sees goes through `L("English", "Русский")` from `astra/i18n.py`
- `ASTRAUTOMA.bat` has to stay ASCII with CRLF, cmd breaks otherwise
- don't commit weights (`*.pth`), `config.json` or logs
- say in the PR what you tested in the game
