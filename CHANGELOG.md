# Changelog

## 1.2.0 — 2026-09-22

- **Every moon.** Minmus, Ike, Gilly, Laythe, Vall, Tylo, Bop and Pol — landing or low orbit,
  flown as a route: out of one sphere of influence, across, down into the next.
- **Return home from any mission.** In *Mission target* press **R**: the plan and the Δv budget get
  the way back — lift-off, leaving the moon, the transfer home, straight into Kerbin's air, parachutes.
  **Return home** in the menu brings any craft back from wherever it is right now.
- **Rendezvous and docking.** The app lists every craft in orbit of every body; pick one and it
  flies there, matches planes, finds the intercept with the game's own orbit prediction, stops next
  to it and, if you ask, docks port to port on RCS (the thrusters are calibrated by itself).
- **Gravity assists.** Trips to other planets try a flyby of every other planet on exact Lambert
  arcs; the flyby is used only when it saves at least 50 m/s (Moho via Eve −790 m/s, Eeloo via
  Jool −685 m/s). Weights 1.1 ship a 20-year atlas of the best assists.
- Leaving a moon for another planet no longer dives to a low parking orbit first (Tylo → Kerbin
  4.6 km/s instead of 6.4).
- Fixed: capture at any body other than Kerbin, Mun, Minmus or Duna stopped with an error.
- The new skills are unlocked by weights **1.1**, released on their own.

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
