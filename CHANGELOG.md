# Changelog

## 1.1.0 — 2026-09-22

- **The autopilot does not give up.** A failed step is retried twice; then you choose:
  retry, take manual control, or let it rescue the craft (stable orbit, parachutes or a
  powered landing). No answer in 60 s — it rescues the craft by itself.
- **"Next" hint** on the main menu: it tells you what to do now, and the cursor waits on it.
- **AI setup in one key.** On the first start the app finds `ASTRAUTOMA-Weights-*.zip` in
  Downloads or on the Desktop and installs it. In *Weights* press **A** to install everything,
  **I** to import a downloaded archive.
- Release page: the program (`ASTRAUTOMA-vX.zip`) and the AI (`ASTRAUTOMA-Weights-*.zip`) are
  separate, clearly named files. The launcher keeps its Windows line endings in the zip.
- Website, checks on every push, contributing guide, code of conduct and security policy.

## 1.0.0 — 2026-09-21

First public version.

- Pre-flight analysis: Δv, fuel, TWR, control, power, comms, legs — green / yellow / red.
- Automatic flight to low and high Kerbin orbit, low and high Mun orbit and a Mun landing,
  with a live plan table, PC-clock ETAs and ±Δt drift.
- Interplanetary targets: Moho, Eve, Duna, Dres, Jool and Eeloo — landing, low and high orbit.
- Trained navigator: departure windows, aerocapture, go / no-go, comms relays.
- Relay satellites: how many are needed, delivered one per flight into their own slots.
- Rocket design from real KSP parts: engines, tanks, decouplers, required and optional equipment.
- Weights as separate modules installed from a library — the app ships clean.
- English and Russian interface.
