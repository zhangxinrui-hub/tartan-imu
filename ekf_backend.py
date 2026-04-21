"""
ekf_backend.py
--------------
Error-State Kalman Filter (ES-EKF) for IMU orientation estimation.
Replaces GT quaternion dependency in TartanIMU evaluation pipeline.

State definition
────────────────
  Nominal state : q  (scipy Rotation, world-from-body)
                  bg (gyro bias in body frame, rad/s)
  Error state   : δx = [δθ (3), δbg (3)]  ∈ R^6
  Covariance    : P  ∈ R^{6×6}

Predict step  – driven by gyroscope at IMU rate
  q  ← q ⊗ Rot(ω_c · dt)          ω_c = gyro − bg
  bg ← bg                          (random walk)
  P  ← F P Fᵀ + Q

Update step   – accelerometer as gravity sensor (corrects pitch/roll only)
  h(q) = R(q)⁻¹ · g_world         predicted gravity in body frame
  ν    = a_meas − h(q)             residual
  H    = [skew(h(q))  |  0₃ₓ₃]    Jacobian wrt δθ, δbg
  K    = P Hᵀ (H P Hᵀ + Rₐ)⁻¹
  q    ← q ⊗ Rot(δθ),  bg ← bg + δbg
  P    ← (I−KH)P(I−KH)ᵀ + K Rₐ Kᵀ  (Joseph form)

Yaw is observable only through gyro integration (no magnetometer),
so yaw drift accumulates.  This is expected and documented.

Usage
─────
    from ekf_backend import ESKF, run_eskf_sequence
    quats = run_eskf_sequence(acc_raw, gyro_raw, q0, bg0, dt)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix such that skew(a)·b = a×b."""
    v = np.asarray(v, dtype=np.float64)
    return np.array(
        [[ 0.0,   -v[2],  v[1]],
         [ v[2],   0.0,  -v[0]],
         [-v[1],   v[0],  0.0 ]],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────
# ES-EKF class
# ─────────────────────────────────────────────────────────────

class ESKF:
    """
    Error-State Kalman Filter for orientation + gyro-bias estimation.

    Parameters
    ----------
    q0 : array [4] (x,y,z,w) or scipy Rotation
        Initial orientation (world-from-body).
    bg0 : array [3]
        Initial gyro bias estimate (rad/s).
    std_gyro : float
        Gyroscope white-noise standard deviation (rad/s per sample).
    std_bg : float
        Gyro bias random-walk std (rad/s² / sqrt(Hz) · √dt).
    std_acc : float
        Accelerometer noise std used in gravity update (m/s²).
    acc_gate : float
        Accept accelerometer gravity update only when
        |‖a‖ − 9.81| / 9.81 < acc_gate.  Rejects dynamic phases.
    """

    def __init__(
        self,
        q0,
        bg0,
        std_gyro: float = 2e-3,
        std_bg:   float = 1e-5,
        std_acc:  float = 0.5,
        acc_gate: float = 0.25,
    ):
        if isinstance(q0, R):
            self.q = q0
        else:
            q0_arr = np.asarray(q0, dtype=np.float64)
            self.q = R.from_quat(q0_arr / np.linalg.norm(q0_arr))

        self.bg = np.asarray(bg0, dtype=np.float64).copy()

        # Error-state covariance (6×6)
        self.P = np.eye(6) * 1e-6

        self.std_gyro = float(std_gyro)
        self.std_bg   = float(std_bg)
        self.std_acc  = float(std_acc)
        self.acc_gate = float(acc_gate)

        # Pre-built identity matrix
        self._I6 = np.eye(6)

    # ----------------------------------------------------------
    def predict(self, gyro: np.ndarray, dt: float) -> None:
        """
        Propagate nominal state and covariance using gyroscope.

        gyro : [3] rad/s  (raw measurement, bias not yet removed)
        dt   : float      seconds between samples
        """
        omega = gyro - self.bg          # bias-corrected angular velocity

        # Nominal quaternion integration (right perturbation in body frame)
        self.q = self.q * R.from_rotvec(omega * dt)

        # Linearized transition matrix F (6×6)
        #   δθ_new  = (I − [ω]ₓ·dt)·δθ − I·dt·δbg
        #   δbg_new = δbg
        F = self._I6.copy()
        F[:3, :3] -= skew(omega) * dt
        F[:3,  3:] = -np.eye(3) * dt

        # Discrete process noise Q
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (self.std_gyro ** 2)
        Q[3:, 3:] = np.eye(3) * (self.std_bg   ** 2) * dt

        self.P = F @ self.P @ F.T + Q

    # ----------------------------------------------------------
    def update_acc(self, acc: np.ndarray) -> bool:
        """
        Correct orientation using accelerometer as gravity sensor.
        Only pitch & roll are observable; yaw remains gyro-only.

        acc  : [3] m/s²  raw accelerometer (must still contain gravity,
                         i.e. gravity NOT yet subtracted)
        Returns True if the update was applied, False if gated out.
        """
        a_norm = float(np.linalg.norm(acc))
        if a_norm < 1e-6:
            return False

        # Gate: reject update when linear acceleration dominates
        if abs(a_norm - 9.81) / 9.81 > self.acc_gate:
            return False

        # Expected gravity direction in body frame (from nominal q)
        g_world    = np.array([0.0, 0.0, 9.81])
        g_body_nom = self.q.inv().apply(g_world)   # [3]

        # Measured gravity direction (scale to 9.81)
        g_body_meas = acc * (9.81 / a_norm)

        # Residual
        nu = g_body_meas - g_body_nom              # [3]

        # Jacobian H = [skew(g_body_nom) | 0₃ₓ₃]
        # Derivation: h(q⊗Rot(δθ)) ≈ g_body_nom + skew(g_body_nom)·δθ
        H = np.zeros((3, 6))
        H[:3, :3] = skew(g_body_nom)

        # Measurement noise
        R_acc = np.eye(3) * (self.std_acc ** 2)

        # Kalman gain
        S = H @ self.P @ H.T + R_acc
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(3)).T   # (6,3)

        # Error-state estimate
        dx     = K @ nu                            # [6]
        dtheta = dx[:3]
        dbg    = dx[3:]

        # Apply correction to nominal state
        self.q   = self.q * R.from_rotvec(dtheta)
        self.bg += dbg

        # Covariance update – Joseph form for numerical stability
        IKH     = self._I6 - K @ H
        self.P  = IKH @ self.P @ IKH.T + K @ R_acc @ K.T

        return True

    # ----------------------------------------------------------
    def get_quat(self) -> np.ndarray:
        """Return current orientation as [x, y, z, w] array."""
        return self.q.as_quat()

    def get_rotation(self) -> R:
        """Return current orientation as scipy Rotation."""
        return self.q


# ─────────────────────────────────────────────────────────────
# Convenience wrapper: run over an entire sequence
# ─────────────────────────────────────────────────────────────

def run_eskf_sequence(
    acc_raw:      np.ndarray,
    gyro_raw:     np.ndarray,
    q0,
    bg0:          np.ndarray,
    dt:           float,
    use_acc:      bool  = True,
    std_gyro:     float = 2e-3,
    std_bg:       float = 1e-5,
    std_acc:      float = 0.5,
    acc_gate:     float = 0.25,
) -> np.ndarray:
    """
    Run the ES-EKF over a full IMU sequence.

    Parameters
    ----------
    acc_raw   : [N, 3] m/s²  raw accelerometer (WITH gravity)
    gyro_raw  : [N, 3] rad/s raw gyroscope     (WITH bias)
    q0        : initial orientation ([x,y,z,w] or scipy Rotation)
    bg0       : [3]   initial gyro bias estimate
    dt        : float  sample period (seconds)
    use_acc   : bool   whether to apply accelerometer gravity update

    Returns
    -------
    quats : [N, 4] float64  estimated quaternions in [x,y,z,w] order
    """
    N = len(acc_raw)
    quats = np.zeros((N, 4), dtype=np.float64)

    ekf = ESKF(q0, bg0, std_gyro=std_gyro, std_bg=std_bg,
               std_acc=std_acc, acc_gate=acc_gate)

    for k in range(N):
        ekf.predict(gyro_raw[k], dt)
        if use_acc:
            ekf.update_acc(acc_raw[k])
        quats[k] = ekf.get_quat()

    return quats


# ─────────────────────────────────────────────────────────────
# Stage 2B: Velocity-aided ES-EKF
# ─────────────────────────────────────────────────────────────

def estimate_orientation_from_gravity(acc_window: np.ndarray) -> R:
    """
    Estimate initial orientation from gravity only.
    Pitch/roll are observable, yaw is set implicitly to zero.
    """
    acc_mean = np.mean(np.asarray(acc_window, dtype=np.float64), axis=0)
    if np.linalg.norm(acc_mean) < 1e-8:
        return R.identity()
    g_world = np.array([[0.0, 0.0, 9.81]], dtype=np.float64)
    rot, _ = R.align_vectors(g_world, acc_mean.reshape(1, 3))
    return rot


def sigma_to_std(
    sigma_proxy: np.ndarray,
    sigma_scale: float = 0.15,
    sigma_floor: float = 0.05,
    clamp_min: float = -4.0,
    clamp_max: float = 4.0,
) -> np.ndarray:
    """
    Convert model uncertainty proxy to a positive velocity std.

    output_block2 values are small signed numbers; we treat them as
    log-scale confidence proxies and map them to positive std with:

        std = sigma_floor + sigma_scale * exp(clamp(proxy))
    """
    sigma_proxy = np.asarray(sigma_proxy, dtype=np.float64)
    sigma_proxy = np.clip(sigma_proxy, clamp_min, clamp_max)
    return sigma_floor + sigma_scale * np.exp(sigma_proxy)


class VelocityAidedESKF:
    """
    Error-state EKF with IMU propagation and neural velocity updates.

    Nominal state
    -------------
      p  : position in world frame
      v  : velocity in world frame
      q  : orientation (world-from-body)
      bg : gyro bias in body frame
      ba : accelerometer bias in body frame

    Error-state
    -----------
      dx = [dp, dv, dtheta, dbg, dba] ∈ R^15
    """

    def __init__(
        self,
        q0=None,
        p0=None,
        v0=None,
        bg0=None,
        ba0=None,
        std_gyro: float = 2e-3,
        std_acc: float = 2e-2,
        std_bg: float = 1e-5,
        std_ba: float = 1e-4,
        gravity: np.ndarray | None = None,
    ):
        self.p = np.zeros(3, dtype=np.float64) if p0 is None else np.asarray(p0, dtype=np.float64).copy()
        self.v = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.bg = np.zeros(3, dtype=np.float64) if bg0 is None else np.asarray(bg0, dtype=np.float64).copy()
        self.ba = np.zeros(3, dtype=np.float64) if ba0 is None else np.asarray(ba0, dtype=np.float64).copy()
        self.q = R.identity() if q0 is None else (q0 if isinstance(q0, R) else R.from_quat(np.asarray(q0, dtype=np.float64)))

        self.P = np.eye(15, dtype=np.float64) * 1e-3
        self.std_gyro = float(std_gyro)
        self.std_acc = float(std_acc)
        self.std_bg = float(std_bg)
        self.std_ba = float(std_ba)
        self.g = np.array([0.0, 0.0, 9.81], dtype=np.float64) if gravity is None else np.asarray(gravity, dtype=np.float64)
        self._I15 = np.eye(15, dtype=np.float64)

    def predict(self, gyro: np.ndarray, acc: np.ndarray, dt: float) -> None:
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        omega = gyro - self.bg
        f_b = acc - self.ba

        # Nominal state propagation
        a_world = self.q.apply(f_b) - self.g
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = self.q * R.from_rotvec(omega * dt)

        # Linearized error propagation
        Rwb = self.q.as_matrix()
        F = self._I15.copy()
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -Rwb @ skew(f_b) * dt
        F[3:6, 12:15] = -Rwb * dt
        F[6:9, 6:9] -= skew(omega) * dt
        F[6:9, 9:12] = -np.eye(3) * dt

        Q = np.zeros((15, 15), dtype=np.float64)
        Q[3:6, 3:6] = np.eye(3) * (self.std_acc ** 2) * dt * dt
        Q[6:9, 6:9] = np.eye(3) * (self.std_gyro ** 2) * dt * dt
        Q[9:12, 9:12] = np.eye(3) * (self.std_bg ** 2) * dt
        Q[12:15, 12:15] = np.eye(3) * (self.std_ba ** 2) * dt

        self.P = F @ self.P @ F.T + Q

    def update_velocity(
        self,
        vel_body_meas: np.ndarray,
        sigma_proxy: np.ndarray | None = None,
        axes: np.ndarray | list[int] | None = None,
        fixed_std: float | np.ndarray | None = None,
        sigma_scale: float = 0.15,
        sigma_floor: float = 0.05,
    ) -> np.ndarray:
        """
        Update using neural body-frame velocity measurement.

        z = v_hat_body
        h(x) = R(q)^T * v_world
        """
        vel_body_meas = np.asarray(vel_body_meas, dtype=np.float64).reshape(3)
        if axes is None:
            axes = np.array([0, 1, 2], dtype=int)
        else:
            axes = np.asarray(axes, dtype=int)

        vel_body_nom = self.q.inv().apply(self.v)
        Rbw = self.q.inv().as_matrix()

        H_full = np.zeros((3, 15), dtype=np.float64)
        H_full[:, 3:6] = Rbw
        H_full[:, 6:9] = skew(vel_body_nom)

        z = vel_body_meas[axes]
        z_hat = vel_body_nom[axes]
        H = H_full[axes]

        if fixed_std is not None:
            std = np.asarray(fixed_std, dtype=np.float64)
            if std.ndim == 0:
                std = np.full(len(axes), float(std))
            else:
                std = std[axes]
        else:
            sigma_proxy = np.zeros(3, dtype=np.float64) if sigma_proxy is None else np.asarray(sigma_proxy, dtype=np.float64).reshape(3)
            std = sigma_to_std(sigma_proxy, sigma_scale=sigma_scale, sigma_floor=sigma_floor)[axes]

        R_meas = np.diag(std ** 2)
        nu = z - z_hat
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(len(axes))).T
        dx = K @ nu

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = self.q * R.from_rotvec(dx[6:9])
        self.bg += dx[9:12]
        self.ba += dx[12:15]

        IKH = self._I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        return std

    def get_state(self):
        return {
            'p': self.p.copy(),
            'v': self.v.copy(),
            'q': self.q.as_quat().copy(),
            'bg': self.bg.copy(),
            'ba': self.ba.copy(),
            'P': self.P.copy(),
        }


def run_velocity_ekf_sequence(
    acc_raw: np.ndarray,
    gyro_raw: np.ndarray,
    vel_body_meas: np.ndarray,
    vel_sigma_proxy: np.ndarray | None,
    dt: float,
    q0=None,
    p0=None,
    v0=None,
    bg0=None,
    ba0=None,
    axes: np.ndarray | list[int] | None = None,
    fixed_std: float | np.ndarray | None = None,
    sigma_scale: float = 0.15,
    sigma_floor: float = 0.05,
    std_gyro: float = 2e-3,
    std_acc: float = 2e-2,
    std_bg: float = 1e-5,
    std_ba: float = 1e-4,
) -> dict:
    """
    Run velocity-aided EKF over a full sequence.

    acc_raw / gyro_raw / vel_body_meas are expected to have the same length N.
    """
    acc_raw = np.asarray(acc_raw, dtype=np.float64)
    gyro_raw = np.asarray(gyro_raw, dtype=np.float64)
    vel_body_meas = np.asarray(vel_body_meas, dtype=np.float64)
    vel_sigma_proxy = None if vel_sigma_proxy is None else np.asarray(vel_sigma_proxy, dtype=np.float64)
    N = len(acc_raw)

    if q0 is None:
        q0 = estimate_orientation_from_gravity(acc_raw[: min(200, N)])
    if bg0 is None:
        bg0 = np.mean(gyro_raw[: min(200, N)], axis=0)
    if ba0 is None:
        ba0 = np.zeros(3, dtype=np.float64)

    ekf = VelocityAidedESKF(
        q0=q0,
        p0=p0,
        v0=v0,
        bg0=bg0,
        ba0=ba0,
        std_gyro=std_gyro,
        std_acc=std_acc,
        std_bg=std_bg,
        std_ba=std_ba,
    )

    pos_hist = np.zeros((N, 3), dtype=np.float64)
    vel_hist = np.zeros((N, 3), dtype=np.float64)
    quat_hist = np.zeros((N, 4), dtype=np.float64)
    vel_std_hist = np.zeros((N, 3), dtype=np.float64)

    for k in range(N):
        ekf.predict(gyro_raw[k], acc_raw[k], dt)
        sigma_k = None if vel_sigma_proxy is None else vel_sigma_proxy[k]
        std_used = ekf.update_velocity(
            vel_body_meas[k],
            sigma_proxy=sigma_k,
            axes=axes,
            fixed_std=fixed_std,
            sigma_scale=sigma_scale,
            sigma_floor=sigma_floor,
        )
        state = ekf.get_state()
        pos_hist[k] = state['p']
        vel_hist[k] = state['v']
        quat_hist[k] = state['q']
        vel_std_hist[k, np.asarray(axes if axes is not None else [0, 1, 2], dtype=int)] = std_used

    return {
        'pos': pos_hist,
        'vel': vel_hist,
        'quat': quat_hist,
        'vel_std': vel_std_hist,
    }
