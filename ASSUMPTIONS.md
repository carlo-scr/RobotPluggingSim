# Modelling assumptions (v0.2)

We have no CAD, gripper geometry or kinematics for WR1 yet, so the sim is built on **strong,
explicit placeholder assumptions**. Every number below is a field in `plugsim/config.py`. Change it
there (or pass `--cfg '{"section.field": value}'` to `scripts/generate_dataset.py`) and everything
downstream follows. The right-hand column says what real information would replace each guess.

## Robot

| Assumption | Value | Config | Replace with |
|---|---|---|---|
| Arms | 2 × 7-DoF, **Franka-Panda kinematics**, capsule geometry | `scene._PANDA_*` | WR1 URDF / DH parameters |
| Mounting | bases at (0, ±0.26, 0.80) m, upright; torso + head behind | `arm.base_*` | WR1 shoulder poses |
| Base | parked (not simulated) during plugging | (none) | base pose logs if it moves |
| Home pose | TCP ≈ (0.36, ∓0.25, 1.06), pointing down; left = mirror of right | `arm.home_q` | WR1 ready pose |
| Joint control | position PD (Panda gains), gravity compensated | `arm.kp/kv` | WR1 actuator model (real actions are torques) |
| Gripper | parallel jaw, 80 mm stroke, **force-controlled 40 N**, pad μ 1.5; 4 mm fingers with a 20 mm gripping pad, then stepped 2.5 mm outwards (to clear plugs in the row above) | `gripper.*` | WR1 gripper specs |
| Cameras | 7 of WR1's 8 (head front, chest L/R, 2 per gripper; head-back omitted), 384×216 @ control rate; gripper cams named `front`/`rear` (hand ±x side) | `camera.*`, `scene._arm_xml` | real mounting poses, intrinsics, names (`below`/`above`) |

## Task and scene

| Assumption | Value | Config | Replace with |
|---|---|---|---|
| Connector | RJ45 8P8C plug: body 11.7×8.0×21 mm + 16 mm boot, 12 g, **always latch-up** | `plug.*` | other connector types (SFP, LC fibre), orientations |
| Jack | channel = plug + **1.0 mm clearance**, 20 mm deep, **1.5 mm × 45° lead-in**; each side one convex wedge mesh | `port.*`, `scene._jack_assets` | measured jack geometry (real clearance ≈ 0.2–0.5 mm) |
| Latch | detent up to 10 N over the last 5 mm, then a weld at 15 mm depth if aligned (≤1.2 mm, ≤5°). Latched plugs can't be pulled out | `port.latch_*`, `seat_depth` | measured insertion-force curve, unplug tasks |
| Rack | 19" rack; **2 × 1U 24-port patch panels** (18.5 mm pitch, numbered label strips), face at x ≈ 0.53 m, rows around z = 1.05 m; the central ±12 cm of ports are physical jacks (possible targets), the outer ones visual-only | `rack.*` | real rack layouts |
| Rack furniture | top-of-rack 48-port switch, servers, blanking panels, cable managers (rings visual-only), neighbouring racks, raised floor, cable tray, ceiling lights | `scene._rack_furniture_xml`, `_background_xml` | site photos |
| Rack pose | nominal ±1 cm / ±1° (the base never parks identically); OOD `rack_offset` 2–5 cm / 3–5° | `scenario.rack_pose` | localisation error stats |
| Occupancy | 0–50 % of the other ports hold colour-coded patch cables (collidable plug + boot, visual-only drooping cable); OOD `dense_ports` 85–100 % | `scenario.occupied` | real panel fill levels |
| Goal spec | target port marked by a **blue LED** (+ index in annotations); OOD `led_distractors` lights others | `scene._panel_xml` | how the technician actually knows the target |
| Lighting | key + fill + 2 ceiling lights + headlight, ±20 % / slight tint; OOD dim / harsh spot / coloured / flickering | `scenario.lights`, `headlight`, `flicker` | site lighting |
| Clutter | 0–1 object on the shelf (≥14 cm from the cable), 25 % chance of a loose second cable; OOD `clutter`, `distractor_cable` | `scenario.clutter`, `loose_cable` | — |
| Sensors | mild noise (≤2/255) and gain jitter; OOD camera noise, motion blur, overexposure, lens occlusion, camera shift | `scenario.image_fx`, `env.apply_image_fx` | real camera specs |
| Cable | Cat6, 0.70 m, 20 capsule links with ball-joint bending springs, 45 g/m | `cable.*` | measured stiffness/length |
| Cable far end | plugged into a "switch" on the shelf (fixed anchor), side random | `cable.anchor_*` | real cable routing |
| Start state | plug + cable **lying on a shelf** in front of the rack, smooth random layout | `scenario.sample_layout` | hanging / coiled / handed-over starts |
| Collisions | cable collides with world, plug, fingers and occupant plugs, **not** with arm links, itself, cable-manager rings or occupant cables | `scene` contype bits | enable if snagging matters |

## Control and demonstrations

| Assumption | Value | Where |
|---|---|---|
| Action | 16-D: per arm TCP Δpos (m), Δrot (rotvec), grip ∈ [0,1], compliant ∈ {0,1}, at 20 Hz | `env.py` |
| Compliant mode | IK gains (1, .1, .1, .3, .3, .3) about a fixed tool tip 31 mm ahead of the TCP | `gripper.compliant_gains/tool_tip` |
| Demonstrator | privileged scripted expert (one arm, the one on the plug's side) with closed-loop re-measurement, servo, and back-off + spiral retries | `expert.py` |
| Nominal noise | speed ×0.85–1.2, grasp jitter σ 1.5 mm, port-pose noise σ 0.3 mm / 0.3° | `expert.py` |
| Policy demos | DART noise σ 2.5 mm/step (+ rotation) in free-space phases; label = expert's clean action (`action.expert` column) | `datagen.DART_PHASES`, `--dart` |
| Learned policy | Diffusion Policy, x0-prediction, chunks of 16 relative TCP targets (8 executed), state or image observations | `learn/policy.py` |
| Observation | 44-D proprio (q, qd, TCP pose, finger width per arm) + camera frames | `env.observe` |

Real WR1 data is 52-D state / 41-D torque-plus-gripper actions (`GaugeFailClassification`'s config
says 47/31), with 8 cameras at 60 Hz. Mapping between the two will need the real layout.

## Physics numerics (found while tuning, keep in mind)

* MuJoCo uses **max(friction)** over a contact pair, so plug/jack μ is 0.3. A grippy plug makes the
  45° chamfer self-locking.
* Plug↔jack contacts use sharper `solimp` (`port.solimp`). Each jack side is a single convex
  wedge: separate chamfer/wall/face boxes leave internal corners the plug edge catches on.
  Contact softness still widens the funnel by ≈0.5 mm (funnel ≈ ±2 mm at 1 mm clearance).
* The controller runs between `mj_step1` and `mj_step2`, and kinematics are refreshed after every
  control step, so `get_state()`/`set_state()` + replaying actions is **bit-exact**.
* External disturbances (`bump`) act on the hand, not on the 12 g plug, to keep the sim stable.
  Unstable episodes are flagged (`unstable`, `failure_mode = sim_unstable`).
