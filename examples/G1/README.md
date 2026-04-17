# Finetuning GR00T N1.7 on Unitree G1 (SONIC VLA format)

This example finetunes GR00T N1.7 on datasets produced by the
[GR00T-WholeBodyControl](https://github.com/NVIDIA/GR00T-WholeBodyControl)
`gear_sonic` converter — specifically
`gear_sonic/scripts/convert_isaac_hdf5_to_lerobot.py`, which ingests
robomimic-layout HDF5 rollouts from Isaac Lab and emits a LeRobot v2.1
dataset with the SONIC VLA feature schema (ego-view RGB, full G1 joint
proprio, VR 3-point teleop commands, planner commands, per-hand finger
targets).

## Architecture note — why the VLA predicts teleop commands, not joint targets

The SONIC stack is a two-stage controller:

1. **VLA (this finetune)**: vision + language + robot state → teleop command stream (VR 3-point wrist/torso pose, finger joints, wrist-twist joints, planner_*).
2. **Downstream WBC controller** (not trained here): teleop commands → full-body joint torques.

Accordingly, `action.wbc` in the dataset — the full-body joint target — is
**not** a VLA output. The converter populates it from the rollout's recorded
`robot0_joint_pos` as a fallback only, and this config excludes it from the
action modality. See the `script_config["notes"]` block written into
`meta/info.json` by the converter for the authoritative statement.

## Dataset prerequisites

Your dataset must be produced by the `gear_sonic` converter at its current
schema (v2, converter_version 1). The converter writes `meta/modality.json`
via `gear_sonic.data.features_sonic_vla.get_modality_config_sonic_vla`; the
state/action sub-keys referenced by [`gear_sonic_config.py`](gear_sonic_config.py)
must exist in that file. If you regenerate the dataset with a different joint
group layout, update the `state.modality_keys` list here to match.

## Finetuning

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/your/gear_sonic_dataset \
  --modality-config-path examples/G1/gear_sonic_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/g1_sonic_finetune
```

> **If you change the action horizon** (`delta_indices=list(range(0, N))`) in
> [`gear_sonic_config.py`](gear_sonic_config.py), you must regenerate
> `meta/relative_stats.json` by re-running
> `python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag NEW_EMBODIMENT`.
> See [`getting_started/data_config.md`](../../getting_started/data_config.md).

## Open-loop evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /path/to/your/gear_sonic_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/g1_sonic_finetune/checkpoint-<N> \
  --traj-ids 0 \
  --action-horizon 16 \
  --modality-keys vr_3pt_position vr_3pt_orientation left_hand_joints right_hand_joints
```

## What's included vs. excluded

**State** (`observation.*` sub-keys from `meta/modality.json`):

| Included | Excluded | Reason |
|---|---|---|
| `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand` | | Joint proprio (sin/cos encoded) |
| `left_wrist_pos/abs_quat`, `right_wrist_pos/abs_quat` | | Current Cartesian wrist pose — paired with `vr_3pt_*` action targets |
| `root_orientation`, `projected_gravity` | | Base attitude |
| | `cpp_rotation_offset` | Zero-filled by converter |
| | `init_base_quat` | Static per-episode — no per-step signal |

**Action** (`teleop.*` / `action.*` sub-keys):

| Included | Excluded | Reason |
|---|---|---|
| `vr_3pt_position`, `vr_3pt_orientation` | | Primary teleop targets |
| `left_hand_joints`, `right_hand_joints` | | Finger targets — VR can't capture fingers |
| `left_wrist_joints`, `right_wrist_joints` | | Wrist-twist disambiguation |
| `planner_movement`, `planner_facing`, `planner_speed`, `planner_height` | | Continuous base-motion commands |
| | `action.wbc` | Full-body joint target is a downstream controller output, not a VLA output |
| | `action.motion_token`, `teleop.smpl_joints/pose/body_quat_w/target_body_orientation` | Zero-filled by converter (v1) |
| | `planner_mode`, `stream_mode` | int32 discrete flags — LeRobot's `compute_episode_stats` skips integer dtypes, so GR00T's regression action head has no normalization stats for them |
| | `delta_heading` | Zero-filled constant in converter |

All included action keys are registered as
`ABSOLUTE` / `NON_EEF` / `DEFAULT`. `NON_EEF` because the 3-point VR pose
(3×position + 3×rot6d) does not fit GR00T's 9-D EEF contract (1 position + 1
rot6d); flattening into `DEFAULT` avoids a shape hack.
