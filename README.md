# mujoco-arm

Language-driven pick-and-place with a 6-DOF UR5e + Robotiq 2F-85 in MuJoCo.
The roadmap ends in a port to NVIDIA Isaac Lab.

## Setup

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/menagerie
python3 -m venv .venv-linux                    # Windows: python -m venv .venv
.venv-linux/bin/pip install -r requirements.txt
.venv-linux/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # only for fk_ik_exercise_collab.py
```

## Run

```bash
.venv-linux/bin/python demo_reach.py             # 3D viewer   (Windows: .venv/Scripts/python)
.venv-linux/bin/python demo_reach.py --headless  # text report only
.venv-linux/bin/python fundamentals/dh_viewer.py # DH frames in 3D
.venv-linux/bin/python fk_ik_exercise_collab.py  # FK / Jacobian / IK drill (runs on CPU or CUDA)
```

Viewer tips: double-click a body to select it, Ctrl + right-drag to push it, Tab hides the panels.

## Files

| File | What it does |
|---|---|
| `scene.py` | Builds the scene: loads the arm and gripper from MuJoCo Menagerie, attaches them with `MjSpec`, adds floor, cube and box |
| `ik.py` | Damped least-squares inverse kinematics, written from scratch (~40 lines of logic) |
| `ik.py: solve_ik_multistart` | Retries IK from other starting poses when it gets stuck; rejects poses where the arm hits the floor or box |
| `demo_reach.py` | Plans IK waypoints, runs them as smooth joint motion, and checks the result after each segment (with one re-grasp) |
| `eval_place.py` | Step 2b: many headless trials with random cube poses → success rate (95% CI), failure types, grasp/place margins, CSV and plots; `--repeats`, `--yaw-sweep` |
| `plot_results.py` | Draws `results/<tag>/outcomes.png` (start position and cube rotation per outcome) |
| `render_trials.py` | Re-runs chosen trials and saves them as GIFs (single, or original vs fixed side by side) |

## Checkpoint 1 roadmap

- [x] Step 0: scene + arm + gripper
- [x] Step 1: own IK solver, reach and grasp
- [x] Step 2: place the cube in the box (carry high → lower → release → retreat)
- [x] Step 2b: make it robust: random cube poses, 100+ headless trials, find and fix what fails (see below)
- [ ] Step 3: wrap motions as skills: `pick(obj)`, `place(target)`
- [ ] Step 4: text → LLM (tool calling / JSON) → skill calls
- [ ] Step 5: voice → Whisper → step 4

## Things to experiment with (and understand why)

- `damping` in `solve_ik`: set it to 0 and send the arm to the edge of its reach. What happens?
- Remove the world-frame rotation in `ik.py:pose_error`. Why does IK stop converging?
- Remove the smoothstep in `demo_reach.py`. Watch the cube slip.
- Change the cube mass or friction until the grasp fails. That's the real-world gripper problem.

## Step 2b results: robustness evaluation

```bash
.venv-linux/bin/python eval_place.py --baseline   # original planner
.venv-linux/bin/python eval_place.py              # with the fixes below
.venv-linux/bin/python eval_place.py --workspace wide --pos-noise 5 --yaw-noise 5 --rand-physics  # hardest
.venv-linux/bin/python eval_place.py --baseline --workspace wide --replay 7   # watch one failure in 3D
```

Each trial samples a cube start position and yaw (uniform over 360°). Variants add perception
noise (Gaussian error on the cube pose the planner sees, standing in for a camera), random cube
physics (edge 3.6–6 cm, mass 50–300 g, friction 0.4–1.2) and a wider workspace. Every trial is
classified: `OK`, `IK_FAIL`, `GRASP_MISS` (cube didn't rise on lift), `DROPPED` (left the hand while
carrying) or `MISSED_BOX`.

| Variant (100 trials, seed 0) | Baseline | Fixed |
|---|---|---|
| default workspace | 100% | 100% |
| + 5 mm / 5° perception noise | 94% | 100% |
| + random cube physics | 100% | 100% |
| wide workspace | 92% | 100% |
| wide + noise + random physics | 81% | 100% |
| **held-out: wide + noise + physics, 500 trials, seed 1** | **80%** | **100%** |

Where it breaks: perception noise sweep, default workspace, same noise in mm and degrees.

| Noise (mm / deg) | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| Baseline | 100% | 94% | 71% | 53% | 38% | 25% |
| Fixed | 100% | 100% | 100% | 81% | 62% | 47% |

So a camera pipeline for this gripper needs better than about 10 mm and 10° of pose accuracy,
or the grasp has to close the loop (look again and correct) instead of trusting one estimate.

Is it the pose or the noise? `--repeats 10` runs each of 100 poses with 10 noise samples
(15 mm / 15°): 86% OK overall (95% CI 84–88%), **no pose always fails, 76 of 100 are fragile**
(sometimes OK, sometimes not). At this noise level failure is driven by the perception error, not
by where the cube is. The re-grasp rescued 229 of the 366 missed grasps.

![original vs fixed planner, trial 7](results/gifs/trial7_compare.gif)

```bash
.venv-linux/bin/python eval_place.py --pos-noise 15 --yaw-noise 15 --repeats 10
.venv-linux/bin/python render_trials.py --workspace wide --trial 7 --compare
.venv-linux/bin/python render_trials.py --pos-noise 15 --yaw-noise 15 \
    --fragile results/fixed_default_noise15mm15deg_x10/poses.csv --limit 2
```

### What this test does *not* cover (it is still an idealized setup)

- **Control:** each joint is a MuJoCo position servo (PD, kp 2000 / kd 400 on the big joints,
  kp 500 / kd 100 on the wrist, torque limits) with no gravity compensation or feedforward.
  It lags: the TCP ends a segment up to 34 mm from the IK target and grasps with ~5–7 mm error.
  There is no motor model, latency, backlash, joint friction or sensor noise.
- **Perception:** ground truth plus optional Gaussian noise, read once per attempt.
- **Objects:** one cube, always flat (only the yaw is random).
- **Model:** the simulated robot matches the "real" one exactly (no sim-to-real gap).

### Margin, not just pass/fail: the yaw sweep

Pass/fail is deterministic per trial and hides how close a run came to failing. Every trial
also logs three continuous measurements, and the summary prints mean ± std:

- `grasp_offset_mm`: cube centre vs fingertip centre right after closing
- `grasp_twist_deg`: how far closing the fingers rotated the cube (signed, so read the std)
- `place_error_mm`: cube distance from the box centre at the end (OK runs)

`--yaw-sweep` keeps the cube at one spot (x 0.45, y 0.20) and steps its yaw through one
90° period, for both planners:

```bash
.venv-linux/bin/python eval_place.py --yaw-sweep                       # no noise, 1° steps
.venv-linux/bin/python eval_place.py --yaw-sweep --pos-noise 5 --yaw-noise 5 --repeats 10 --sweep-step 3
```

| Yaw sweep | Original planner | Fixed planner |
|---|---|---|
| No noise: OK | 91/91 | 91/91 |
| No noise: grasp offset | 8.5 ± 4.7 mm (max 19.4) | 4.7 ± 0.6 mm (max 5.7) |
| No noise: grasp twist | ± 13.4° (max 27.5°) | ± 0.0° |
| 5 mm / 5° noise, 10 samples per yaw: OK | 293/310 (95%) | 310/310 (100%) |

![yaw sweep with noise](results/yawsweep_default_noise5mm5deg_x10/yaw_sweep.png)

Without noise both planners pass every yaw, but between ~15° and ~75° the original planner
shoves the cube up to 19 mm sideways and twists it up to 27° while closing: the 2F-85 drags it
square. With noise, its failures land exactly in that band. The margin predicted where it
would break before anything failed. The ~5 mm offset that remains for the fixed planner is the
joint PD controller sagging (see below), not the planner.

Held-out run (500 trials, hardest setting): grasp offset 8.8 ± 9.5 mm (max 108) original vs
5.9 ± 4.0 mm (max 20) fixed; placement error 9.7 ± 4.6 mm vs 7.6 ± 3.2 mm.

### What failed and how it was fixed

1. **Fingers closing on the cube's corners** (`GRASP_MISS` at yaw near ±45°). Without noise the
   2F-85 twists the cube square as it closes, so this stayed hidden until perception noise was
   added. Fix: turn the gripper to match the cube's yaw. A cube repeats every 90°, so the
   wrist turns at most ±45° (`grasp_angles`).
2. **IK stuck in local minima** (`IK_FAIL` in the wide workspace, mostly on "carry to box").
   The targets were all reachable: DLS wound the base joint the long way round into its ±360°
   limit. Fix: `solve_ik_multistart` retries from home and random poses and keeps the converged
   answer closest to the current pose.
3. **IK solutions that go through the table.** A restart once found a valid gripper pose
   with the forearm in the floor. Fix: reject solutions where the arm touches the floor or box
   (`robot_hits_scene`), and allow restarts only for the two big moves at safe height. Short
   moves near the table must continue smoothly from the previous pose; otherwise the joint
   interpolation sweeps the arm through the scene.
4. **Cube right next to the box wall.** The aligned grasp put a finger 17 mm into the wall.
   Fix: also try the other grasp (fingers on the other pair of faces, 90° turned).
5. **Closing the loop.** After "lift" the cube height is checked. On a miss the gripper
   opens, backs off, looks again and re-plans once. In the held-out run this saved 1 trial in 500.

Lessons: a 100% score in the easy setting hid two failure modes, and only randomization
found them. Every fix came from looking at a specific failed trial (`--replay`), not guessing.
