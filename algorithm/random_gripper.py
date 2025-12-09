import numpy as np
import pybullet as p  # used only for quaternion/Euler conversion

# =============== Utility Functions ===============

def _sample_point_on_upper_hemisphere(center, radius, z_min=None):
    """
    Sample a point on the upper hemisphere centered at `center` with radius `radius`.
    If z_min is provided, ensure sampled point has z >= z_min.
    """
    center = np.asarray(center, dtype=float)
    while True:
        u = np.random.rand()
        theta = np.arccos(u)               # [0, π/2]
        phi = 2 * np.pi * np.random.rand() # [0, 2π)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        pos = center + np.array([x, y, z])

        if (z_min is None) or (pos[2] >= z_min):
            return pos


def _perturb_direction(direction, max_angle_rad):
    """
    Add random angular perturbation to a direction within a cone of max_angle_rad.
    """
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)

    r = np.random.randn(3)
    r /= np.linalg.norm(r)

    # ensure random vector is not parallel to direction
    if abs(np.dot(r, d)) > 0.99:
        r = np.array([d[1], -d[0], 0.0])
        if np.linalg.norm(r) < 1e-8:
            r = np.array([0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)

    # Orthonormal basis d, v, w
    v = np.cross(d, r)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)

    # Random direction within cone
    a = np.random.rand() * max_angle_rad      
    b = np.random.rand() * 2 * np.pi          

    new_dir = (np.cos(a) * d +
               np.sin(a) * (np.cos(b) * v + np.sin(b) * w))
    return new_dir


def _rotation_matrix_to_quat(R):
    """Convert a 3×3 rotation matrix to a quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def _quat_from_forward_and_roll(forward, max_roll_rad):
    """
    Build a quaternion from a forward direction + random roll.
    The resulting orientation satisfies:
        local Z -> forward
        local X,Y chosen to keep 'up' close to world up.
    """
    f = np.array(forward, dtype=float)
    f /= np.linalg.norm(f)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(world_up, f)
    right /= np.linalg.norm(right)
    up = np.cross(f, right)

    # roll ∈ [-max_roll, max_roll]
    roll = (np.random.rand() * 2.0 - 1.0) * max_roll_rad
    c = np.cos(roll)
    s = np.sin(roll)

    right_rolled = c * right + s * up
    up_rolled = -s * right + c * up

    R = np.column_stack([right_rolled, up_rolled, f])
    return _rotation_matrix_to_quat(R)


# =============== Public API =================

def generate_random_gripper_pose(
    cube_center,
    radius=0.3,
    table_z=0.0,
    min_clearance=0.12,
    max_angle_deg=10,
    max_roll_deg=180,
):
    """
    Generate a random gripper pose:
    - Position sampled from upper hemisphere around the object.
    - Forward direction aimed roughly at the object with angular noise.
    - Additional random roll around the forward axis.

    Returns:
        (px, py, pz, rx, ry, rz) where rotations are Euler angles in radians.
    """
    cube_center = np.asarray(cube_center, dtype=float)
    z_min = table_z + min_clearance

    # 1. Position: upper hemisphere + height constraint
    g_pos = _sample_point_on_upper_hemisphere(cube_center, radius, z_min=z_min)

    # 2. Ideal forward = direction from gripper to cube
    ideal_forward = cube_center - g_pos

    # 3. Randomly perturb direction within a cone (completely random, no filtering)
    max_angle_rad = np.deg2rad(max_angle_deg)
    noisy_forward = _perturb_direction(ideal_forward, max_angle_rad)

    # 4. Add limited roll around the forward axis
    max_roll_rad = np.deg2rad(max_roll_deg)
    quat = _quat_from_forward_and_roll(noisy_forward, max_roll_rad)

    # 5. Convert to Euler angles (xyz), unit: radians
    euler = p.getEulerFromQuaternion(quat)  # (rx, ry, rz)

    px, py, pz = g_pos.tolist()
    rx, ry, rz = euler
    return px, py, pz, rx, ry, rz