# Robust pick-and-place in MuJoCo, driven by evaluation

A 6-DOF UR5e arm with a Robotiq 2F-85 gripper picks a cube and places it in a box, in the
[MuJoCo](https://mujoco.org) physics simulator. The inverse kinematics is written from scratch.
The focus is **measuring** how robust the system is, then fixing what the measurements show.

![original planner vs improved planner on the same trial](results/gifs/trial7_compare.gif)

*Same cube pose, two planners. Left: the original planner's IK gets stuck in a local minimum.
Right: the improved planner restarts IK, checks for collisions and completes the task.*

**Results**

- **80% → 100% success on 500 held-out trials** in the hardest setting: wide workspace,
  5 mm / 5° perception noise, random cube size, mass and friction.
- Breaking point found: the system holds 100% up to 10 mm / 10° of perception error and
  degrades beyond that. That sets an accuracy target for the camera pipeline.
- Continuous grasp measurements predicted *where* the original planner would fail
  before any run actually failed.

## How the system works

```
cube pose ──► planner ──► IK ──► smooth joint trajectory ──► joint PD servos ──► MuJoCo physics
(sim or noisy)  waypoints   damped least squares   (smoothstep)       (Menagerie actuators)
                                    ▲                                              │
                                    └──────── checks after each segment ◄──────────┘
                                               (did the cube lift? still in hand?)
```

| Component | What it does |
|---|---|
| [`ik.py`](ik.py) | Damped least-squares IK from scratch; multi-start with collision rejection |
| [`demo_reach.py`](demo_reach.py) | Planner: grasp selection, IK waypoints, segment checks, one re-grasp on a miss |
| [`eval_place.py`](eval_place.py) | Randomized evaluation: success rate with 95% CI, failure types, margins, sweeps |
| [`render_trials.py`](render_trials.py) | Replays any trial from its seed as a GIF (also side by side) |
| [`scene.py`](scene.py) | Builds the scene with `MjSpec`: UR5e + 2F-85 from MuJoCo Menagerie, cube, box |
| [`fundamentals/`](fundamentals) | DH-parameter visualizers; FK/Jacobian/IK derivation drill in PyTorch |

## Evaluation

Each trial samples a cube position and a yaw over the full 360°. Optional variations:
perception noise (Gaussian error on the pose the planner sees), random cube physics (edge
3.6–6 cm, mass 50–300 g, friction 0.4–1.2) and a wider workspace. Each trial is deterministic
and reproducible from its seed. It is classified as `OK`, `IK_FAIL`, `GRASP_MISS`, `DROPPED` or
`MISSED_BOX`, and logs grasp offset, grasp twist and placement error.

| Setting (100 trials) | Original | Improved |
|---|---|---|
| Default workspace | 100% | 100% |
| + 5 mm / 5° perception noise | 94% | 100% |
| + random cube physics | 100% | 100% |
| Wide workspace | 92% | 100% |
| Wide + noise + physics | 81% | 100% |
| **Held out: wide + noise + physics, 500 trials, new seed** | **80%** (CI 76–83%) | **100%** (CI 99–100%) |

![outcomes of the original planner in the hardest setting](results/baseline_wide_noise5mm5deg_randphys/outcomes.png)

*Original planner, hardest setting. IK failures sit at the edges of the workspace. Most grasp
misses happen where the cube is rotated about 30–45° (right panel).*

### Breaking point: perception noise

| Noise (mm / deg) | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| Original | 100% | 94% | 71% | 53% | 38% | 25% |
| Improved | 100% | 100% | 100% | 81% | 62% | 47% |

At 15 mm / 15°, each of 100 poses was run with 10 noise samples. **No pose always failed, and
76 of 100 were fragile**, so failures come from perception error, not from where the cube is.
The re-grasp step rescued 229 of 366 missed grasps.

### Margin, not just pass/fail: yaw sweep

The cube stays at one spot while its yaw steps through one 90° period. Without noise, both
planners pass every yaw. Between about 15° and 75°, though, the original planner shoves the cube
up to 19 mm and twists it up to 27° while closing; the gripper drags it square. Once noise is
added, its failures land exactly in that band.

![grasp offset and twist vs cube yaw](results/yawsweep_default_noise5mm5deg_x10/yaw_sweep.png)

| Yaw sweep | Original | Improved |
|---|---|---|
| Grasp offset, no noise | 8.5 ± 4.7 mm (max 19.4) | 4.7 ± 0.6 mm (max 5.7) |
| Grasp twist, no noise | ± 13.4° (max 27.5°) | ± 0.0° |
| Success, 5 mm / 5° noise, 10 samples per yaw | 293/310 | 310/310 |

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

## Limitations

This is still an idealized setup, and the limits are stated here on purpose:

- **Control:** each joint is a position servo (PD, kp 2000 / kd 400 on the large joints) with
  no gravity compensation. It lags by up to 34 mm during motion and grasps with about 5 mm of
  error. There is no motor model, latency, backlash or sensor noise.
- **Perception:** ground truth plus Gaussian noise, read once per attempt.
- **Objects:** one rigid cube, always flat (only the yaw varies).
- **No sim-to-real gap:** the simulated robot is the "real" robot.

## Reproduce

```bash
git clone https://github.com/mahesaputranata/mujoco-manipulation && cd mujoco-manipulation
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/menagerie
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python demo_reach.py                                    # watch one pick-and-place in 3D
.venv/bin/python eval_place.py --baseline                         # original planner, 100 trials
.venv/bin/python eval_place.py --workspace wide --pos-noise 5 --yaw-noise 5 --rand-physics
.venv/bin/python eval_place.py --yaw-sweep --pos-noise 5 --yaw-noise 5 --repeats 10 --sweep-step 3
.venv/bin/python eval_place.py --baseline --workspace wide --replay 7   # replay one trial in 3D
.venv/bin/python render_trials.py --workspace wide --trial 7 --compare  # ... or as a GIF
```

Run `eval_place.py` with no flags for the improved planner, and `--baseline` for the original.
100 trials take about 15 s on 8 CPU cores. Results go to `results/<tag>/`.

## Roadmap

The end goal is a real 6-DOF arm on a Jetson that folds cloth from a voice command, using a
vision-language-action (VLA) policy.

- [x] Scene, damped least-squares IK from scratch, pick-and-place
- [x] Randomized evaluation and robustness fixes (this page)
- [ ] Replace the joint PD with gravity compensation, then an operational-space / impedance
      controller, and re-run this evaluation to compare
- [ ] Camera-based cube pose estimation (target: under 10 mm / 10° error, from the sweep above)
- [ ] Cloth in simulation; teleoperated demonstrations; imitation-learning policy (ACT / diffusion)
- [ ] Real arm and Jetson; VLA fine-tuning (e.g. NVIDIA GR00T); voice commands

## Acknowledgements

The robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(UR5e, Robotiq 2F-85), under their respective licenses. They are not included in this repo.
The code here is under the [MIT License](LICENSE).
