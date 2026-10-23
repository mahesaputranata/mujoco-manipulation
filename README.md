# mujoco-arm

Language-driven pick-and-place with a 6-DOF UR5e + Robotiq 2F-85 in MuJoCo.
The roadmap ends in a port to NVIDIA Isaac Lab.

## Run

```bash
.venv/Scripts/python demo_reach.py            # 3D viewer
.venv/Scripts/python demo_reach.py --headless # text report only
```

Viewer tips: double-click a body to select it, Ctrl + right-drag to push it, Tab hides the panels.

## Files

| File | What it does |
|---|---|
| `scene.py` | Builds the scene: loads the arm and gripper from MuJoCo Menagerie, attaches them with `MjSpec`, adds floor, cube and box |
| `ik.py` | Damped least-squares inverse kinematics, written from scratch (~40 lines of logic) |
| `demo_reach.py` | IK waypoints → smooth joint motion → pick the cube and place it in the box |

## Checkpoint 1 roadmap

- [x] Step 0: scene + arm + gripper
- [x] Step 1: own IK solver, reach and grasp
- [x] Step 2: place the cube in the box (carry high → lower → release → retreat)
- [ ] **Step 2b (your exercise): make it robust.** Randomize `CUBE_START` (and cube yaw!),
      run 100 headless trials, report the success rate. Find and fix what fails.
- [ ] Step 3: wrap motions as skills: `pick(obj)`, `place(target)`
- [ ] Step 4: text → LLM (tool calling / JSON) → skill calls
- [ ] Step 5: voice → Whisper → step 4

## Things to experiment with (and understand why)

- `damping` in `solve_ik`: set it to 0 and send the arm to the edge of its reach. What happens?
- Remove the world-frame rotation in `ik.py:pose_error`. Why does IK stop converging?
- Remove the smoothstep in `demo_reach.py`. Watch the cube slip.
- Change the cube mass or friction until the grasp fails. That's the real-world gripper problem.
