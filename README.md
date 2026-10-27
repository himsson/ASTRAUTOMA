<div align="center">

<img src="assets/logo.png" width="320" alt="ASTRAUTOMA">

# ASTRAUTOMA

**An autopilot for Kerbal Space Program that plans the flight, checks your rocket and flies it.**

<a href="README.md"><img src="https://img.shields.io/badge/English-5b8cff?style=for-the-badge" alt="English"></a>
<a href="README.ru.md"><img src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-30363d?style=for-the-badge" alt="Русский"></a>

[![Version](https://img.shields.io/github/v/release/himsson/ASTRAUTOMA?label=version&color=5b8cff)](https://github.com/himsson/ASTRAUTOMA/releases/latest)
![KSP](https://img.shields.io/badge/KSP-1.12-3ddbc4)
![kRPC](https://img.shields.io/badge/kRPC-0.6-3ddbc4)
![Python](https://img.shields.io/badge/Python-3.10%2B-5b8cff)
![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-5b8cff)
[![Checks](https://github.com/himsson/ASTRAUTOMA/actions/workflows/check.yml/badge.svg)](https://github.com/himsson/ASTRAUTOMA/actions/workflows/check.yml)
[![License](https://img.shields.io/badge/license-source--available-f5a524)](LICENSE)
[![Stars](https://img.shields.io/github/stars/himsson/ASTRAUTOMA?style=flat&color=f5a524)](https://github.com/himsson/ASTRAUTOMA/stargazers)


**[🌐 Website](https://himsson.github.io/ASTRAUTOMA/)**

</div>

---

A separate program, not a mod. It talks to the game over kRPC, reads your save and the craft on
the pad, shows what it intends to do, and then does it — ascent, transfer, capture, landing,
rendezvous, docking and the way home.

<p align="center"><img src="docs/flight.png" width="820" alt="Flight table"></p>

## Install

1. KSP 1.12 and the [kRPC](https://github.com/krpc/krpc/releases) mod (`GameData/kRPC`).
2. [Python 3.10+](https://www.python.org/downloads/), with *Add python.exe to PATH* ticked.
3. `ASTRAUTOMA-vX.zip` from the [latest release](https://github.com/himsson/ASTRAUTOMA/releases/latest), unpacked anywhere.
4. The weights — see below. Without them there are no mission targets.
5. In KSP: load a save, open the kRPC window, **Start Server**.
6. Run `ASTRAUTOMA.bat`. Missing Python packages install themselves on first run.

## Weights

The trained part ships separately: `ASTRAUTOMA-Weights-v*.zip` from the
[weights release](https://github.com/himsson/ASTRAUTOMA/releases). Put it in *Downloads* or on
the Desktop and the first start offers to install it.

Later: **Weights / Knowledge → A** installs the newest of everything, **I** imports an archive,
and single modules (flight profile, pilot, designer, mathematician, navigator, knowledge) can be
switched between versions. Weight files are read as plain numbers — code hidden in one cannot run.

## What it does

**Before launch.** Reads the craft and reports Δv, fuel, TWR, control authority, power, comms and
landing gear, each green, yellow or red against what the mission needs. Shows the flight plan step
by step, and next to it what it will do if a step fails — the contingency branches are computed
before launch, because in the air there is no time to compute them.

**In flight.** Launch window, heading corrected for the planet's rotation, gravity turn,
circularization, the transfer phase angle from where the bodies actually are. Every step carries
an arrival time on your clock and how far ahead or behind it is running. When something changes,
the plan is rewritten and the change is marked.

**Destinations.** Every planet and moon of the stock system — Moho, Eve, Duna, Dres, Jool, Eeloo,
Minmus through Tylo. Aerobraking where there is air, parachutes where they help, gravity assists
when a flyby of Eve or Jool is cheaper than flying direct. **Return home** brings a craft back from
wherever it is.

**Rendezvous and docking.** Pick any craft in orbit of any body: it flies there, matches orbits
and docks.

**Comms.** If a probe would lose signal behind a planet, it says so and delivers relay satellites
first — as many as the orbit needs, each to its own slot.

**Design and building.** Sizes the rocket from the Δv budget, picks real parts from your GameData
and builds the craft in the VAB: stages, decouplers, boosters, fairing, equipment. The stage plan
follows what people actually build — learned from 497 real craft.

**Failures.** A dead engine is detected from the thrust reading in about a second and a half. What
happens next depends on the phase: during ascent and landing the prepared branch runs immediately,
in orbit you get 30 seconds to take over. Crew first, hardware second — the rescue counts the
people on board and says so in the verdict.

**Your world.** Rescaled systems, CommNet settings and career limits are read from the save. Mod
parts are picked up from GameData.

<p align="center">
  <img src="docs/analysis.png" width="49%" alt="Pre-flight analysis">
  <img src="docs/design.png" width="49%" alt="Rocket design">
</p>

## Benchmarks

One exam, the same for every version of the weights, run without the game by
`tools/bench_designer.py` and `tools/benchmark.py` in the
[training repository](https://github.com/himsson/ASTRAUTOMA-Training).

| Version | Piloting | Design | Planning | Emergencies |
|---|---|---|---|---|
| **1.x** | 100.0 % | 84.2 % | 97.8 % | 4.0 % |
| **2.0** | — | — | — | — |

**Piloting** — 40 ascents in the hardest simulator, from the pad, at full difficulty: how many
reach a stable orbit. **Design** — 190 projects (38 targets × 5 kinds of craft): how many end as a
`.craft` the game's loader accepts. **Planning** — 9 216 missions: how often the navigator picks
the right route, window and assist. **Emergencies** — crewed flights where the hardware breaks
mid-ascent: how many end with the crew alive.

Releases 1.1, 1.2 and 1.3 carry the same networks byte for byte — one training run, packed three
times — so they share a row. The 4 % under Emergencies is honest: those weights were never shown a
failure, and a pilot that has not seen one flies into the ground — the
few survivors got there by luck, not by handling it. That is what 2.0 is for.

Maths is not in the table. Hohmann, vis-viva, phase angles and Lambert have exact answers, and the
program computes them with the formulas in `rocket_math`; the neural mathematician that ships in
the weights is not used in flight.

## Questions

**Is it a mod?** No. Only kRPC goes into `GameData`.

**Does it work with mods?** Yes — planet data comes from the running game, parts from your
GameData.

**My rocket failed.** Open an [issue](https://github.com/himsson/ASTRAUTOMA/issues) with the flight
table and the newest file from `logs/`. Every flight is logged in full.

## License

[Source-available](LICENSE), not open source. Read the code, use the program, fork it to send a
pull request. Redistributing it, publishing a fork as its own project, or selling it or anything
built on it is not allowed.
