from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Modality config for GR00T-WholeBodyControl "SONIC VLA" datasets on the
# Unitree G1, produced by gear_sonic/scripts/convert_isaac_hdf5_to_lerobot.py.
#
# Supervised VLA targets are the teleop command stream (VR 3-point pose,
# per-hand finger joints, per-wrist joint trio, planner_* base commands),
# NOT the full-body WBC joint target — action.wbc is populated in the dataset
# as a fallback sourced from joint_pos and is intentionally excluded here.
# Zero-filled signals (action.motion_token, teleop.smpl_*, teleop.body_quat_w,
# teleop.target_body_orientation) are also excluded.

gear_sonic_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            # Proprioceptive joint groups (in radians — good candidates for sin/cos).
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            # Current wrist Cartesian pose (directly comparable to vr_3pt_* action).
            "left_wrist_pos",
            "left_wrist_abs_quat",
            "right_wrist_pos",
            "right_wrist_abs_quat",
            # Base attitude.
            "root_orientation",
            "projected_gravity",
            # Excluded: cpp_rotation_offset (zero-filled by converter),
            # init_base_quat (static per episode — carries no per-step info).
        ],
        sin_cos_embedding_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "vr_3pt_position",
            "vr_3pt_orientation",
            "left_hand_joints",
            "right_hand_joints",
            "left_wrist_joints",
            "right_wrist_joints",
            "planner_mode",
            "planner_movement",
            "planner_facing",
            "planner_speed",
            "planner_height",
            # Excluded: action.wbc (full-body joint target — produced downstream
            # by the WBC controller, not the VLA);
            # stream_mode / delta_heading (constants or zero-filled in the
            # stationary-pick converter output — no learning signal).
        ],
        action_configs=[
            # vr_3pt_position: 9D = 3× (x,y,z) for (left wrist, right wrist, torso),
            # in world frame. Doesn't fit the 9D EEF contract (which is 1 point +
            # rot6d), so NON_EEF/DEFAULT and let the model treat it as a flat vector.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # vr_3pt_orientation: 18D = 3× rot6d. Same story — 3 points, so can't
            # be declared EEF/XYZ_ROT6D. Flat DEFAULT.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Finger targets (7D canonical finger representation per hand).
            # ABSOLUTE matches how SO-100 treats its gripper and avoids drift
            # on discrete-ish open/close motions.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Wrist roll/pitch/yaw (3D per side). Exists in the teleop stream
            # because the 3-point VR pose can't disambiguate wrist twist around
            # the forearm axis. ABSOLUTE (no matching 3D wrist-joint state key
            # to use as a RELATIVE reference).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Planner commands — high-level base motion. Constants for the
            # stationary-pick task in the v1 converter, but included so this
            # config generalizes to locomotion datasets without a rewrite.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(gear_sonic_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
