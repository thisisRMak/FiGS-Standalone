"""
Synthesize IMU measurements from FiGS tXUd trajectory arrays.

Converts the 10D state + 4D control trajectory into accelerometer and
gyroscope readings suitable for visual-inertial odometry (e.g., OpenVINS).

tXUd format (15+ rows x N columns):
  Row 0:      time (s)
  Row 1-3:    position (px, py, pz)
  Row 4-6:    velocity (vx, vy, vz)
  Row 7-10:   quaternion (qx, qy, qz, qw)
  Row 11:     normalized thrust (uf)
  Row 12-14:  body rates (wx, wy, wz) in rad/s

Output IMU CSV:
  timestamp, ax, ay, az, wx, wy, wz
  (accel in m/s^2, gyro in rad/s)

Usage:
  python -m figs.utilities.imu_synthesizer <tXUd.npy> -o imu.csv [--noise euroc]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# EuRoC ADIS16448 IMU noise parameters (matches OpenVINS defaults)
NOISE_PRESETS = {
    "euroc": {
        "accel_noise_density": 2.0e-3,   # m/s^2/sqrt(Hz)
        "accel_random_walk": 3.0e-3,     # m/s^3/sqrt(Hz)
        "gyro_noise_density": 1.6968e-4,  # rad/s/sqrt(Hz)
        "gyro_random_walk": 1.9393e-5,   # rad/s^2/sqrt(Hz)
    },
    "none": {
        "accel_noise_density": 0.0,
        "accel_random_walk": 0.0,
        "gyro_noise_density": 0.0,
        "gyro_random_walk": 0.0,
    },
}

# Default drone parameters (carl.json)
DEFAULT_MASS = 1.144
DEFAULT_FORCE_NORMALIZED = 6.90
DEFAULT_N_ROTORS = 4


def quat_to_rotmat(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix (body-to-world)."""
    return Rotation.from_quat(q).as_matrix()


def synthesize_imu(
    tXUd,
    mass=DEFAULT_MASS,
    force_normalized=DEFAULT_FORCE_NORMALIZED,
    n_rotors=DEFAULT_N_ROTORS,
    imu_rate=200.0,
    noise_preset="euroc",
    seed=None,
):
    """Synthesize IMU measurements from a FiGS tXUd trajectory.

    The tXUd trajectory is typically at 20 Hz.  VIO systems need IMU data
    at a much higher rate (100-400 Hz) so that there are many IMU samples
    between consecutive camera frames.  This function upsamples by linearly
    interpolating thrust (uf) and gyroscope readings to ``imu_rate``.

    Args:
        tXUd: (15+, N) trajectory array.
        mass: Drone mass in kg.
        force_normalized: Per-motor normalized force gain (fn from frame config).
        n_rotors: Number of rotors (tn = force_normalized * n_rotors).
        imu_rate: Output IMU sample rate in Hz (default 200).
        noise_preset: Noise profile name ("euroc", "none") or dict with noise params.
        seed: Random seed for reproducibility.

    Returns:
        imu_data: (M, 7) array — [timestamp, ax, ay, az, wx, wy, wz].
    """
    if isinstance(noise_preset, str):
        noise = NOISE_PRESETS[noise_preset]
    else:
        noise = noise_preset

    rng = np.random.default_rng(seed)

    # Extract state and controls at tXUd rate
    t_key = tXUd[0]                # (N,) timestamps
    vel_key = tXUd[4:7].T          # (N, 3) world-frame velocity
    quat_key = tXUd[7:11].T        # (N, 4) quaternion [qx, qy, qz, qw]
    gyro_key = tXUd[12:15].T       # (N, 3) body rates [wx, wy, wz]

    # Compute world-frame acceleration via finite differences on velocity
    N_key = len(t_key)
    dt_key = np.diff(t_key)
    vdot_key = np.diff(vel_key, axis=0) / dt_key[:, None]  # (N-1, 3)
    # Pad last sample by repeating
    vdot_key = np.vstack([vdot_key, vdot_key[-1:]])  # (N, 3)

    # Compute specific force in FiGS body frame (FRD): a_body = R^T @ (v_dot - g)
    # FiGS world frame is z-down (NED-like), so g = [0, 0, +9.81]
    g_world = np.array([0.0, 0.0, 9.81])
    R_key = Rotation.from_quat(quat_key).as_matrix()  # (N, 3, 3) body-to-world
    specific_force_world = vdot_key - g_world  # (N, 3)
    accel_body_key = np.einsum('nij,nj->ni', np.transpose(R_key, (0, 2, 1)), specific_force_world)

    # Build high-rate timeline
    dt_imu = 1.0 / imu_rate
    t_imu = np.arange(t_key[0], t_key[-1] + dt_imu * 0.5, dt_imu)
    M = len(t_imu)

    # Interpolate body-frame accel and gyro to IMU rate
    accel_body_imu = np.column_stack([
        np.interp(t_imu, t_key, accel_body_key[:, ax]) for ax in range(3)
    ])
    gyro_imu = np.column_stack([
        np.interp(t_imu, t_key, gyro_key[:, ax]) for ax in range(3)
    ])

    # FiGS uses FRD (Forward-Right-Down) body convention where body z points
    # toward gravity.  VIO systems (OpenVINS) expect FLU (Forward-Left-Up)
    # where body z points UP and accel reads +9.81 at hover.
    # Convert via 180° rotation about x: negate y and z.
    accel = accel_body_imu.copy()
    accel[:, 1] *= -1  # negate y: FRD → FLU
    accel[:, 2] *= -1  # negate z: FRD → FLU

    # Gyro: also flip y and z for FRD → FLU
    gyro_imu[:, 1] *= -1
    gyro_imu[:, 2] *= -1

    # Add IMU noise (at the upsampled rate)
    if noise["accel_noise_density"] > 0:
        accel_white = noise["accel_noise_density"] / np.sqrt(dt_imu)
        accel += rng.normal(0, accel_white, accel.shape)

    if noise["accel_random_walk"] > 0:
        accel_bias_sigma = noise["accel_random_walk"] * np.sqrt(dt_imu)
        accel_bias = np.cumsum(rng.normal(0, accel_bias_sigma, accel.shape), axis=0)
        accel += accel_bias

    if noise["gyro_noise_density"] > 0:
        gyro_white = noise["gyro_noise_density"] / np.sqrt(dt_imu)
        gyro_imu = gyro_imu + rng.normal(0, gyro_white, gyro_imu.shape)

    if noise["gyro_random_walk"] > 0:
        gyro_bias_sigma = noise["gyro_random_walk"] * np.sqrt(dt_imu)
        gyro_bias = np.cumsum(rng.normal(0, gyro_bias_sigma, gyro_imu.shape), axis=0)
        gyro_imu = gyro_imu + gyro_bias

    return np.column_stack([t_imu, accel, gyro_imu])


def save_imu_csv(imu_data, output_path):
    """Save IMU data to CSV file.

    Args:
        imu_data: (N, 7) array — [timestamp, ax, ay, az, wx, wy, wz].
        output_path: Output file path.
    """
    header = "timestamp,ax,ay,az,wx,wy,wz"
    np.savetxt(output_path, imu_data, delimiter=",", header=header,
               fmt="%.9f", comments="# ")


def generate_image_timestamps(image_dir, output_path, image_rate=10.0,
                              start_time=0.0):
    """Generate image_timestamps.csv from a directory of numbered images.

    Assumes images are named sequentially (e.g., 000.png, 001.png, ...).

    Args:
        image_dir: Directory containing images.
        output_path: Output CSV path.
        image_rate: Camera frame rate in Hz.
        start_time: Timestamp of first image.
    """
    image_dir = Path(image_dir)
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(ext))
    images = sorted(images, key=lambda p: p.name)

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    dt_cam = 1.0 / image_rate
    with open(output_path, "w") as f:
        f.write("# timestamp,filename\n")
        for i, img_path in enumerate(images):
            t = start_time + i * dt_cam
            f.write(f"{t:.9f},{img_path.name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize IMU data from FiGS tXUd trajectory"
    )
    parser.add_argument("tXUd_path", help="Path to tXUd .npy file")
    parser.add_argument("-o", "--output", default="imu.csv",
                        help="Output IMU CSV path")
    parser.add_argument("--noise", default="euroc",
                        choices=list(NOISE_PRESETS.keys()),
                        help="IMU noise preset")
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS,
                        help="Drone mass (kg)")
    parser.add_argument("--force-normalized", type=float,
                        default=DEFAULT_FORCE_NORMALIZED,
                        help="Per-motor normalized force gain (fn)")
    parser.add_argument("--n-rotors", type=int, default=DEFAULT_N_ROTORS,
                        help="Number of rotors (tn = fn * n_rotors)")
    parser.add_argument("--imu-rate", type=float, default=200.0,
                        help="Output IMU rate in Hz (default 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise")
    parser.add_argument("--frame-config", type=str, default=None,
                        help="Path to frame config JSON (overrides mass/thrust)")

    args = parser.parse_args()

    # Load frame config if provided
    mass = args.mass
    force_normalized = args.force_normalized
    n_rotors = args.n_rotors
    if args.frame_config:
        with open(args.frame_config) as f:
            fc = json.load(f)
        mass = fc["mass"]
        force_normalized = fc["force_normalized"]
        n_rotors = fc.get("number_of_rotors", n_rotors)

    # Load trajectory
    tXUd = np.load(args.tXUd_path)
    print(f"Loaded tXUd: shape {tXUd.shape}")
    print(f"  Time range: {tXUd[0, 0]:.3f} - {tXUd[0, -1]:.3f} s")
    print(f"  Duration: {tXUd[0, -1] - tXUd[0, 0]:.3f} s")
    print(f"  Samples: {tXUd.shape[1]}")

    # Synthesize IMU
    imu_data = synthesize_imu(
        tXUd,
        mass=mass,
        force_normalized=force_normalized,
        n_rotors=n_rotors,
        imu_rate=args.imu_rate,
        noise_preset=args.noise,
        seed=args.seed,
    )

    # Save
    save_imu_csv(imu_data, args.output)
    print(f"Saved {len(imu_data)} IMU samples to {args.output}")

    # Print sanity check: |mean accel| should be ~9.81 for level flight
    mean_accel = np.mean(imu_data[:, 1:4], axis=0)
    mean_gyro = np.mean(imu_data[:, 4:7], axis=0)
    print(f"  Mean accel: [{mean_accel[0]:.3f}, {mean_accel[1]:.3f}, {mean_accel[2]:.3f}] m/s^2")
    print(f"  Mean gyro:  [{mean_gyro[0]:.4f}, {mean_gyro[1]:.4f}, {mean_gyro[2]:.4f}] rad/s")


if __name__ == "__main__":
    main()
