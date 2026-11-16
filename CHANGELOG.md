# Changelog

## 1.4.0 (2026-09-22)

- the designer builds the craft in the VAB now (press B in the design result). parts go on real
  attach nodes, it gets checked like the game loader does and saved to Ships/VAB
- boosters are picked automatically (tries 0, 2, 4, 6, 8 and keeps the lightest that works)
- heat shield and chutes for the way home, RCS for docking, RTGs past Duna, fairing on every launch
- fixed a probe core being taken for a fuel tank
- weights 1.3: design school on 497 craft

## 1.3.0 (2026-09-22)

- design school: the designer learned from 66 real craft (42 rockets, stations, rovers, bases,
  probes, landers)
- stages, TWR, dv split and engines now look like what people actually build
- new payload types: rover, base, station, telescope/probe, each with its usual equipment
- return home now goes on the top stage with the landing
- weights 1.2 have the design school

## 1.2.0 (2026-09-22)

- all moons: Minmus, Ike, Gilly, Laythe, Vall, Tylo, Bop, Pol
- return home from any mission (R in Mission target) or from wherever the craft is (menu)
- rendezvous and docking with any craft in orbit, RCS calibrates itself
- gravity assists, used only if they save at least 50 m/s. weights 1.1 have a 20 year atlas
- leaving a moon for another planet doesn't go to a low orbit first anymore (Tylo to Kerbin
  4.6 km/s instead of 6.4)
- fixed capture crashing at anything other than Kerbin, Mun, Minmus or Duna
- new stuff is unlocked by weights 1.1

## 1.1.0 (2026-09-22)

- failed steps are retried twice, then you pick retry / manual control / rescue. no answer in
  60s and it rescues the craft itself
- "Next" hint in the main menu
- weights install from a zip in Downloads or on the Desktop
- program and weights are separate zips in releases
- website, CI checks

## 1.0.0 (2026-09-21)

First release.

- preflight check (dv, fuel, TWR, control, power, comms, legs)
- autopilot to Kerbin and Mun orbits and Mun landing with a live plan table
- Moho, Eve, Duna, Dres, Jool, Eeloo
- navigator for windows, aerocapture, go/no-go and relays
- relay sats
- rocket designer from real parts
- weights as separate modules
- English and Russian
