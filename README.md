# Robust pick-and-place in MuJoCo, driven by evaluation

A 6-DOF UR5e arm with a Robotiq 2F-85 gripper picks a cube and places it in a box, in the
[MuJoCo](https://mujoco.org) physics simulator. The inverse kinematics is written from scratch.
The focus is **measuring** how robust the system is, then fixing what the measurements show.

![original system vs improved system on the same trial](results/gifs/trial7_compare.gif)

*Same cube pose, two systems. Left: the original planner's IK gets stuck in a local minimum.
Right: the improved system restarts IK, checks for collisions and completes the task.*

**Results**

- **80% → 99.8% success (400 → 499 of 500 held-out trials)** in the hardest setting: wide
  workspace, 5 mm / 5° perception noise, random cube size, mass and friction.
- **Grasp precision 4.5 mm → 0.4 mm** without noise, from gravity compensation plus waiting for
  the arm to arrive before closing the gripper.
- Breaking point found: the system holds 99–100% up to 10 mm / 10° of perception error and
  degrades beyond that. Under noise, perception error dominates; the controller barely matters.
- Continuous grasp measurements predicted *where* the original system would fail before any
  run actually failed.

## How the system works

```
cube pose ──► planner ──► IK ──► smooth joint trajectory ──► joint PD + gravity comp ──► MuJoCo physics
(sim or noisy)  waypoints   damped least squares   (smoothstep)       (Menagerie servos)
                                    ▲                                                       │
                                    └──────────── checks after each segment ◄───────────────┘
                              (has the arm arrived? did the cube lift? still in hand?)
```

| Component | What it does |
|---|---|
| [`ik.py`](ik.py) | Damped least-squares IK from scratch; multi-start with collision rejection |
| [`demo_reach.py`](demo_reach.py) | Planner: grasp selection, IK waypoints, wait-to-arrive, segment checks, one re-grasp |
| [`controllers.py`](controllers.py) | Joint PD (Menagerie servo) and PD + gravity compensation |
| [`eval_place.py`](eval_place.py) | Randomized evaluation: success rate with 95% CI, failure types, margins, sweeps |
| [`render_trials.py`](render_trials.py) | Replays any trial from its seed as a GIF (also side by side) |
| [`plot_controllers.py`](plot_controllers.py) | Controller comparison and tracking figures |
| [`scene.py`](scene.py) | Builds the scene with `MjSpec`: UR5e + 2F-85 from MuJoCo Menagerie, cube, box |
| [`fundamentals/`](fundamentals) | DH-parameter visualizers; FK/Jacobian/IK derivation drill in PyTorch |

The **original system** is the first working version: fixed grasp orientation, single-start IK,
plain PD control, and the gripper closing on a fixed schedule. The **improved system** has
every fix below.

## Evaluation

Each trial samples a cube position and a yaw over the full 360°. Optional variations:
perception noise (Gaussian error on the pose the planner sees), random cube physics (edge
3.6–6 cm, mass 50–300 g, friction 0.4–1.2) and a wider workspace. Each trial is deterministic
and reproducible from its seed. It is classified as `OK`, `IK_FAIL`, `GRASP_MISS`, `DROPPED` or
`MISSED_BOX`, and logs grasp offset, grasp twist, placement error and tracking errors.

| Setting (100 trials) | Original | Improved |
|---|---|---|
| Default workspace | 100% | 100% |
| + 5 mm / 5° perception noise | 94% | 100% |
| + random cube physics | 100% | 100% |
| Wide workspace | 92% | 100% |
| Wide + noise + physics | 81% | 100% |
| **Held out: wide + noise + physics, 500 trials, new seed** | **80%** (CI 76–83%) | **99.8%** (CI 99–100%) |

The one held-out failure (trial 225) is a large cube next to the robot base: with 6–7 mm of
perception error, a finger lands on top of the cube instead of beside it.

![outcomes of the original system in the hardest setting](results/baseline_wide_noise5mm5deg_randphys/outcomes.png)

*Original system, hardest setting. IK failures sit at the edges of the workspace. Most grasp
misses happen where the cube is rotated about 30–45° (right panel).*

### Breaking point: perception noise

| Noise (mm / deg) | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| Original | 100% | 94% | 71% | 53% | 38% | 25% |
| Improved | 100% | 100% | 99% | 82% | 65% | 49% |

At 15 mm / 15°, each of 100 poses was run with 10 noise samples: 858/1000 OK (95% CI 83–88%).
**No pose always failed, and 80 of 100 were fragile**, so failures come from perception error,
not from where the cube is. The re-grasp step rescued 229 of 371 missed grasps.

### Margin, not just pass/fail: yaw sweep

The cube stays at one spot while its yaw steps through one 90° period. Without noise, both
systems pass every yaw. Between about 15° and 75°, though, the original system shoves the cube
up to 19 mm and twists it up to 27° while closing; the gripper drags it square. Once noise is
added, its failures land exactly in that band.

![grasp offset and twist vs cube yaw](results/yawsweep_default_noise5mm5deg_x10/yaw_sweep.png)

| Yaw sweep | Original | Improved |
|---|---|---|
| Grasp offset, no noise | 8.5 ± 4.7 mm (max 19.4) | 0.4 ± 0.1 mm (max 0.9) |
| Grasp twist, no noise | ± 13.4° (max 27.5°) | ± 0.0° |
| Success, 5 mm / 5° noise, 10 samples per yaw | 293/310 | 310/310 |

## Control: gravity compensation and waiting to arrive

The joints start as MuJoCo position servos, which act as PD controllers:
`τ = kp·(q_target − q) − kd·q̇`. A PD controller only pushes when there is an error, so to hold
the arm up it **must sag** until `kp·error` equals the gravity torque: 11 mm at the gripper.

**Gravity compensation** adds the model's gravity torque `g(q)` (from recursive Newton–Euler) as
feedforward, sent through the servo as `ctrl = q_target + g(q)/kp`. This keeps the servo's
implicitly integrated damping and its torque limits. Computing the damping explicitly in Python
would put the wrist at the edge of numerical stability (kd·dt/I = 1.99, limit 2).

| Improved planner, 100 trials | PD, closes on schedule | PD + wait | Gravity comp, on schedule | **Gravity comp + wait** |
|---|---|---|---|---|
| Sag when holding still | 11.2 mm | 11.2 mm | 0.1 mm | **0.1 mm** |
| Reach error at grasp height | 5.9 mm | 11.1 mm | 10.9 mm | **1.9 mm** |
| Grasp offset (no noise) | 4.5 ± 1.1 mm | 5.9 ± 1.1 mm | 1.7 ± 0.4 mm | **0.4 ± 0.2 mm** |
| Placement error (no noise) | 5.0 ± 2.1 mm | 5.5 ± 2.5 mm | 1.8 ± 0.6 mm | **0.6 ± 0.3 mm** |
| Grasp offset (10 mm / 10° noise) | 8.0 ± 6.5 mm | 8.3 ± 6.6 mm | 9.6 ± 7.1 mm | 8.9 ± 6.5 mm |
| Success (10 mm / 10° noise) | 100% | 100% | 98% | 99% |
| Extra time per pick-and-place | — | 0.9 s | — | 0.6 s |

![gripper height while coming down to the cube, four control set-ups](results/controllers/tracking.png)

What the measurements showed:

1. **Two errors were cancelling.** Coming down, the PD lags behind the trajectory (gripper too
   high) while gravity pulls it down (too low). With plain PD they happened to cancel. Removing
   the sag exposed the lag: the gripper closed 11 mm above the cube's centre (dashed blue line).
2. **The planner closed the gripper on a fixed schedule**, assuming the arm keeps up. Fix:
   before closing or releasing, hold the target until the gripper is within 2 mm, has stopped
   moving, or 1.5 s have passed. Waiting needs gravity compensation: with plain PD it only
   lets the sag fully develop (solid orange line).
3. **Under perception noise the controller barely matters.** A 0.4 mm-precise arm aimed at the
   wrong spot still misses by the perception error. The next gain is in perception or in
   closing the loop visually, not in control.

![controller comparison](results/controllers/comparison.png)

## What failed, and the fixes

Each fix came from replaying a specific failed trial, not from guessing.

1. **Fingers closing on the cube's corners.** This was hidden without noise, because the gripper
   twists the cube square as it closes. **Fix:** turn the gripper to match the cube's yaw. A cube
   repeats every 90°, so the wrist never turns more than 45°.
2. **IK stuck in local minima** on reachable targets. Damped least squares wound the base joint
   the long way round into its ±360° limit. **Fix:** multi-start IK that keeps the converged
   solution closest to the current pose.
3. **IK solutions through the table.** A restart once put the forearm inside the floor.
   **Fix:** reject poses where the arm touches the floor or box, and allow restarts only for big
   moves at a safe height. Otherwise, interpolating in joint space sweeps the arm through the scene.
4. **Cube next to the box wall.** One finger would have entered the wall. **Fix:** fall back to
   the other grasp, with the fingers on the other pair of faces.
5. **Open-loop grasping.** **Fix:** check that the cube actually lifted; on a miss, back off,
   look again and re-plan once.
6. **Gravity sag and closing before the arm arrives** (see Control). **Fix:** gravity
   compensation, and wait for the arm to arrive before closing or releasing.

## Limitations

This is still an idealized setup, and the limits are stated here on purpose:

- **Control:** joint PD with gravity compensation, but no feedforward for the planned motion, so
  the arm still lags up to ~30 mm during fast moves (waiting hides this at the grasp, at a cost of
  0.6 s). The grasped cube's weight is not compensated. There is no motor model, latency, backlash
  or sensor noise.
- **Perception:** ground truth plus Gaussian noise, read once per attempt.
- **Objects:** one rigid cube, always flat (only the yaw varies).
- **No sim-to-real gap:** the simulated robot is the "real" robot.

## Reproduce

```bash
git clone https://github.com/mahesaputranata/mujoco-manipulation && cd mujoco-manipulation
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/menagerie
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python demo_reach.py                                    # watch one pick-and-place in 3D
.venv/bin/python eval_place.py                                    # improved system, 100 trials
.venv/bin/python eval_place.py --baseline                         # original system
.venv/bin/python eval_place.py --workspace wide --pos-noise 5 --yaw-noise 5 --rand-physics
.venv/bin/python eval_place.py --controller pd --no-arrive        # improved planner, original control
.venv/bin/python eval_place.py --yaw-sweep --pos-noise 5 --yaw-noise 5 --repeats 10 --sweep-step 3
.venv/bin/python eval_place.py --baseline --workspace wide --replay 7   # replay one trial in 3D
.venv/bin/python render_trials.py --workspace wide --trial 7 --compare  # ... or as a GIF
.venv/bin/python plot_controllers.py                              # controller figures (needs the runs above)
```

100 trials take about 20 s on 8 CPU cores. Results go to `results/<tag>/`.

## Roadmap

The end goal is a real 6-DOF arm on a Jetson that folds cloth from a voice command, using a
vision-language-action (VLA) policy.

- [x] Scene, damped least-squares IK from scratch, pick-and-place
- [x] Randomized evaluation and robustness fixes
- [x] Gravity compensation and wait-to-arrive; controller comparison (this page)
- [ ] Computed-torque control (removes the motion lag), then operational-space / impedance
      control for contact-rich tasks (in progress)

## Acknowledgements

The robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(UR5e, Robotiq 2F-85), under their respective licenses. They are not included in this repo.
The code here is under the [MIT License](LICENSE).
