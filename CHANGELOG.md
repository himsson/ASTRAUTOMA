# Changelog

## 1.4.0 — 2026-09-22

- **Builds the craft in the VAB.** In the design result press **B**: every part is placed on its real
  attach node — pod, parachute or nose cone, stages, decouplers, fairing, adapters, boosters with nose
  cones, fins, launch clamps, landing legs, rover wheels, panels, antennas and science. The craft is
  checked like the game loader would, then saved to *Ships/VAB* of your save.
- **Boosters on auto.** The designer tries none, 2, 4, 6 and 8 and keeps the lightest rocket that works.
- **Everything the mission needs:** heat shield and parachutes for the way home, parachutes for any
  crew, RCS and monopropellant for docking, RTGs beyond Duna, fairing on every launch from Kerbin.
- Fixed: a probe core was taken for a fuel tank.
- Weights **1.3**: the design school now learned from **497** craft (207 rockets, 152 probes and
  satellites, 65 stations, 37 landers, 20 rovers, 16 bases) out of 3 700+ downloaded from GitHub,
  the Steam Workshop, the stock game and your saves.

## 1.3.0 — 2026-09-22

- **Design school.** The designer learned from 66 real craft — the Steam Workshop, the stock game and
  your own saves; only craft whose every part exists in the game, no planes. 42 rockets, 5 stations,
  5 rovers, 3 bases, 7 probes and telescopes, 4 landers.
- Rockets now follow what people build: the number of stages for the Δv, liftoff and upper-stage TWR,
  how the Δv is split between stages, the engines people put on each stage, nose cones on boosters,
  fins where they are usual. The result names the closest real rockets.
- **What it carries:** rocket only, rover, surface base, space station, telescope / probe — each with
  the kit such craft really carry (wheels, batteries, panels, antennas, docking ports, lights, science).
- The way home now rides on the top stage together with the landing.
- Weights **1.2** carry the design school (Builder module).

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
