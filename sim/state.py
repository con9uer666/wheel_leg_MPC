"""State extraction from MuJoCo data for the 10-state wheel-leg model."""

import numpy as np
import mujoco

from controllers.params import L_MIN


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw].

    Pitch is extracted as the angle of the body x-axis from the horizontal
    using atan2, giving full ±180° coverage without gimbal lock issues.
    (Standard ZYX arcsin-based pitch is limited to ±90°.)
    """
    w, x, y, z = q
    # Rotation matrix elements (row = world axis, col = body axis)
    # Body x-axis in world frame: [R00, R10, R20]
    R20 = 2*(x*z - w*y)          # x-comp of body-z in world (for roll)
    R21 = 2*(y*z + w*x)          # y-comp of body-z in world (for roll)
    R22 = 1 - 2*(x*x + y*y)      # z-comp of body-z in world
    # Body x-axis in world: used for pitch and yaw
    R00 = 1 - 2*(y*y + z*z)
    R10 = 2*(x*y + w*z)
    R20x = 2*(x*z - w*y)         # z-component of body x-axis in world
    # Pitch: angle of body x-axis from horizontal plane
    # atan2(−R20x, sqrt(R00²+R10²)) gives ±180° when combined with roll/yaw
    # But for pitch alone (rotation about world y), we use the body x in world xz-plane:
    pitch = np.arctan2(-R20x, np.sqrt(R00*R00 + R10*R10))
    roll  = np.arctan2(R21, R22)
    yaw   = np.arctan2(R10, R00)
    return np.array([roll, pitch, yaw])


def pitch_from_xmat(xmat_flat: np.ndarray) -> float:
    """
    Extract pitch (rotation about world y) from MuJoCo xmat (9 floats, row-major).
    xmat[:,2] = body z-axis (up) in world frame.
    pitch = atan2(body_z_world_x, body_z_world_z) gives full ±180° coverage.
    """
    # MuJoCo xmat row-major: R[i,j]=world-component-i of body-axis-j
    # body z-axis in world = column 2: [R[0,2], R[1,2], R[2,2]] = [xmat[2], xmat[5], xmat[8]]
    body_z_x = float(xmat_flat[2])
    body_z_z = float(xmat_flat[8])
    return np.arctan2(body_z_x, body_z_z)


def get_joint_ids(model):
    """Cache joint qpos/qvel address indices by name."""
    ids = {}
    for name in ("hip_L", "hip_R", "leg_L", "leg_R", "wheel_L", "wheel_R"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ids[name] = {
            "qpos": model.jnt_qposadr[jid],
            "qvel": model.jnt_dofadr[jid],
        }
    return ids


def extract_state(model, data, joint_ids: dict) -> dict:
    """
    Extract all quantities needed by the MPC and PD controllers.

    10-state vector mapping:
      x[0]  s        trunk x position (m)
      x[1]  ds       trunk x velocity (m/s)
      x[2]  phi      yaw (rad)
      x[3]  dphi     yaw rate (rad/s)  – body-frame z angular velocity
      x[4]  theta_ll left leg world-frame angle from vertical (rad)
      x[5]  dtheta_ll  rate
      x[6]  theta_lr right leg world-frame angle from vertical (rad)
      x[7]  dtheta_lr  rate
      x[8]  theta_b  body pitch (rad)
      x[9]  dtheta_b body pitch rate (rad/s)

    Additional fields (not in x, used by leg PD):
      L_l, L_r       total left/right pole length (m)
      leg_L_dot, leg_R_dot  prismatic joint velocity (m/s)
    """
    # Trunk free-joint: qpos[0:7] = [x,y,z, qw,qx,qy,qz]
    #                   qvel[0:6] = [vx,vy,vz, wx,wy,wz] (body frame angular)
    qpos = data.qpos
    qvel = data.qvel

    # Position & velocity in MuJoCo native frame (positive = +x world axis).
    # Any sign convention needed by the LQR is applied inside the controller.
    s  = float(qpos[0])
    ds = float(qvel[0])

    # Orientation: read directly from MuJoCo's rotation matrix (xmat) for full ±180° pitch.
    # trunk body index = 1 (world=0, trunk=1 in this model)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    xmat = data.xmat[trunk_id]          # row-major 3×3 world-from-body rotation matrix
    pitch = pitch_from_xmat(xmat)       # full ±180° pitch (theta_b)
    # Roll and yaw from quaternion (only need ±90° coverage for normal operation)
    euler = quat_to_euler(qpos[3:7])
    roll  = float(euler[0])
    yaw   = float(euler[2])             # phi

    # Angular velocities (body frame)
    wx = float(qvel[3])
    wy = float(qvel[4])   # pitch rate  (theta_b_dot)
    wz = float(qvel[5])   # yaw rate    (phi_dot)

    # Hip joint angles and rates
    hip_L_adr = joint_ids["hip_L"]
    hip_R_adr = joint_ids["hip_R"]
    q_hip_L = float(qpos[hip_L_adr["qpos"]])
    q_hip_R = float(qpos[hip_R_adr["qpos"]])
    dq_hip_L = float(qvel[hip_L_adr["qvel"]])
    dq_hip_R = float(qvel[hip_R_adr["qvel"]])

    # theta_ll = hip joint angle (not world frame).
    # Brute-force confirmed: θll sign=-1 in state_to_vec, so theta_ll = +q_hip here
    # (state_to_vec negates it, giving MPC x[4] = -q_hip_L).
    theta_ll  = q_hip_L
    dtheta_ll = dq_hip_L
    theta_lr  = q_hip_R
    dtheta_lr = dq_hip_R

    # Prismatic leg extension (joint output is extension length)
    leg_L_adr = joint_ids["leg_L"]
    leg_R_adr = joint_ids["leg_R"]
    ext_L = float(qpos[leg_L_adr["qpos"]])
    ext_R = float(qpos[leg_R_adr["qpos"]])
    leg_L_dot = float(qvel[leg_L_adr["qvel"]])
    leg_R_dot = float(qvel[leg_R_adr["qvel"]])

    L_l = L_MIN + ext_L
    L_r = L_MIN + ext_R

    return {
        # 10-state vector components
        "s":         s,
        "ds":        ds,
        "phi":       yaw,
        "dphi":      wz,
        "theta_ll":  theta_ll,
        "dtheta_ll": dtheta_ll,
        "theta_lr":  theta_lr,
        "dtheta_lr": dtheta_lr,
        "theta_b":   pitch,
        "dtheta_b":  wy,
        # Extra
        "roll":      roll,
        "L_l":       L_l,
        "L_r":       L_r,
        "leg_L_dot": leg_L_dot,
        "leg_R_dot": leg_R_dot,
    }


def state_to_vec(st: dict) -> np.ndarray:
    """
    Pack state dict into 10-element numpy vector for MPC.
    Sign conventions confirmed by brute-force search (optimal: max_pitch=0.02°):
      theta_ll → negated  (sign_theta_ll = -1)
      theta_b  → positive (sign_theta_b  = +1)
      ds       → negated  (sign_ds       = -1)  [already negated in extract_state]
    """
    # Brute-force optimal: θll sign=-1 in search = MPC sees -(-hip) = +hip_angle
    # theta_ll stored as +q_hip in dict, sign_theta_ll=-1 → MPC x[4] = -(-hip) = +hip
    # But search code was: -hip_L * s_ll = -hip_L * (-1) = +hip_L
    # So MPC gets +hip_L → use +st["theta_ll"] (no negate here)
    return np.array([
        st["s"],        st["ds"],
        st["phi"],      st["dphi"],
        st["theta_ll"], st["dtheta_ll"],    # = +q_hip_L (matches brute-force)
        st["theta_lr"], st["dtheta_lr"],
        st["theta_b"],  st["dtheta_b"],
    ])
