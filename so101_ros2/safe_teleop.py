"""Conservative leader → follower teleop that bypasses lerobot-teleoperate.

    poetry run python -m so101_ros2.safe_teleop

Why this exists
---------------
On this SO-ARM101 setup, lerobot-teleoperate's `connect()` calls
`configure_motors()` which writes `Acceleration=254` and
`Maximum_Acceleration=254` to every motor, then enables torque. That
means the first `send_action` makes all 6 motors draw peak current at
their factory `Max_Torque_Limit=1000`, which collapses the data bus.
We've verified failure rate scales with commanded motion magnitude.

This script:
  • applies Max_Torque_Limit=500 and Acceleration=50 to all 6 follower
    motors before running (motor-current cap that stays well below the
    bus brown-out threshold)
  • bypasses lerobot's driver entirely — talks to the Feetech bus through
    the raw scservo_sdk so configure_motors() never runs
  • clips each step to `MAX_STEP` counts so the follower can never slam
  • retries transient sync_read failures up to RETRIES times
  • on Ctrl-C: disables torque + resets goals = present so the arm is
    safe to leave

Calibration
-----------
This script does NOT apply LeRobot's calibration normalization — it
operates in raw motor counts. Leader counts go straight to follower
goal positions. If your leader and follower were calibrated with their
"middle" positions matching the same physical pose (the standard
calibration flow does this), positions will line up. If they were
calibrated at different physical poses, expect offsets.

If teleop feels offset, re-run `lerobot-calibrate` on each arm while
keeping them in the **same physical pose** during the "move to middle
of range" step — that defines the shared zero reference for both arms.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

from scservo_sdk import (
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
    SCS_HIBYTE,
    SCS_LOBYTE,
)

from so101_ros2.settings import (
    FOLLOWER_ID,
    FOLLOWER_PORT,
    JOINT_NAMES,
    LEADER_ID,
    LEADER_PORT,
)

# LeRobot calibration JSONs — written by `lerobot-calibrate` for each arm
CALIB_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
CALIB_FOLLOWER = CALIB_DIR / "robots"        / "so_follower" / f"{FOLLOWER_ID}.json"
CALIB_LEADER   = CALIB_DIR / "teleoperators" / "so_leader"   / f"{LEADER_ID}.json"

# ── STS3215 registers (addresses match lerobot/motors/feetech/tables.py) ────
ADDR_MAX_TORQUE_LIMIT     = 16   # 2 bytes, EEPROM — caps motor current
ADDR_P_COEFFICIENT        = 21   # 1 byte — proportional gain (higher = stronger hold)
ADDR_D_COEFFICIENT        = 22   # 1 byte — derivative gain (higher = more damping)
ADDR_I_COEFFICIENT        = 23   # 1 byte — integral gain
ADDR_PROTECTION_CURRENT   = 28   # 2 bytes, EEPROM — current threshold for protection trip
ADDR_OVERLOAD_TORQUE      = 36   # 1 byte, EEPROM — torque % held when overload protection trips
ADDR_TORQUE_ENABLE        = 40   # 1 byte, RAM
ADDR_ACCELERATION         = 41   # 1 byte, RAM
ADDR_GOAL_POSITION        = 42   # 2 bytes, RAM
ADDR_PRESENT_POSITION     = 56   # 2 bytes, read-only
ADDR_PRESENT_VELOCITY     = 58   # 2 bytes, read-only — sign-magnitude (sign bit 15)
ADDR_PRESENT_LOAD         = 60   # 2 bytes, read-only — sign-magnitude (sign bit 10)
ADDR_MAXIMUM_ACCELERATION = 85   # 1 byte, EEPROM

IDS = [1, 2, 3, 4, 5, 6]
BAUD = 1_000_000

# ── Tunables ────────────────────────────────────────────────────────
FPS          = 30     # loop rate (Hz). 60Hz blew the per-iteration budget on
                      # this 1Mbps bus (read-leader + read-follower + write-goals
                      # is ~6-9 ms before retries; one retry busts the 16.6 ms
                      # budget and causes a visible jerk). 30Hz matches the
                      # baseline that was demonstrably smooth.
MAX_STEP     = 100    # max counts per iteration (~9°). Lower bound from the
                      # working baseline. Higher values let big leader moves
                      # whip the follower's trajectory generator each tick,
                      # which reads as jitter even when bus comms are clean.
READ_RETRIES = 3      # bus is genuinely marginal — 5-15% raw fail rate.
                      # Retries paper over it to ~0.5% effective fail rate.
                      # Lowering retries exposes the failures as discrete-pulse
                      # behaviour (verified: dropping to 1 took success 99.5%→93%).

# Per-motor settings aligned with upstream LeRobot's so101_follower.configure()
# (src/lerobot/robots/so_follower/so_follower.py:156-170). Body joints stay at
# firmware default Max_Torque_Limit=1000 and Max_Acceleration=254 — no local
# caps. The gripper alone gets reduced limits to prevent burnout. P=16 across
# the board, matching upstream's "lower value to avoid shakiness" comment.
#
# WARNING: this rig's 12V supply has historically been marginal at these
# upstream-default settings (see project notes: bus brown-out under load).
# If `so101-teleop` regresses below ~99% success after this change, the
# previous conservative caps were here for a reason — restore them or use
# `lerobot-teleoperate` which writes the same upstream values.
MOTOR_LIMITS: dict[int, tuple[str, int, int, int, int]] = {
    # (name, max_torque, max_accel, p_gain, d_gain)
    1: ("shoulder_pan",  1000, 254, 16, 32),
    2: ("shoulder_lift", 1000, 254, 16, 32),
    3: ("elbow_flex",    1000, 254, 16, 32),
    4: ("wrist_flex",    1000, 254, 16, 32),
    5: ("wrist_roll",    1000, 254, 16, 32),
    6: ("gripper",        500, 254, 16, 32),
}
# Gripper-only burnout protection (upstream so_follower.py:168-170).
GRIPPER_PROTECTION_CURRENT = 250   # ~50% of max current
GRIPPER_OVERLOAD_TORQUE    = 25    # 25% torque when overloaded

# ── ANSI ────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(s: str) -> str: return s if _TTY else ""
GREEN, RED, YEL, BOLD, DIM, RST = (
    _c("\033[32m"), _c("\033[31m"), _c("\033[33m"),
    _c("\033[1m"),  _c("\033[2m"),  _c("\033[0m"),
)

# ── State ───────────────────────────────────────────────────────────
_f_ph: PortHandler | None = None
_l_ph: PortHandler | None = None
_pkt = None
_cleaned = False


def _cleanup() -> None:
    """Disable follower torque and reset goal=present so arm is safe at rest."""
    global _cleaned
    if _cleaned:
        return
    _cleaned = True
    if _f_ph is not None and _pkt is not None:
        try:
            for sid in IDS:
                _pkt.write1ByteTxRx(_f_ph, sid, ADDR_TORQUE_ENABLE, 0)
                pos, _, _ = _pkt.read2ByteTxRx(_f_ph, sid, ADDR_PRESENT_POSITION)
                _pkt.write2ByteTxRx(_f_ph, sid, ADDR_GOAL_POSITION, pos)
        except Exception:
            pass
    for ph in (_f_ph, _l_ph):
        if ph is not None:
            try:
                ph.closePort()
            except Exception:
                pass


def _on_signal(*_):
    print("\nstopping — torque off, goals reset…", flush=True)
    _cleanup()
    sys.exit(0)


def _patched_setPacketTimeout(self, packet_length):  # noqa: N802
    """Replacement for scservo_sdk's PortHandler.setPacketTimeout.

    The vendor SDK hardcodes LATENCY_TIMER=16ms (FTDI-era assumption) into
    every timeout: timeout = (tx_per_byte * len) + 32ms + 2ms ≈ 34ms minimum.
    On CDC-ACM (cdc_acm driver, no FTDI buffering) realistic latency is
    1-4ms. With READ_RETRIES=3, a single miss costs 4 × 34ms ≈ 136ms of
    loop stall — exactly the discrete-pulse jerk we're seeing.

    Use 6ms slop here: covers normal CDC-ACM latency with margin, and a
    full 4-attempt failure cascade now costs ~26ms (under one 30Hz tick)
    instead of 136ms (over four ticks). Lerobot's feetech driver uses an
    equivalent monkeypatch (src/lerobot/motors/feetech/feetech.py:85).
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + 6.0


def _open(port: str) -> PortHandler:
    ph = PortHandler(port)
    if not ph.openPort():
        raise RuntimeError(f"openPort({port}) failed — permission? in use?")
    if not ph.setBaudRate(BAUD):
        raise RuntimeError(f"setBaudRate({BAUD}) failed on {port}")
    # Monkeypatch the SDK's slow timeout calc — see _patched_setPacketTimeout.
    ph.setPacketTimeout = _patched_setPacketTimeout.__get__(ph, PortHandler)
    return ph


def _disable_leader_torque() -> None:
    """Make sure the leader is hand-movable. Some lerobot code paths leave
    torque enabled on the leader; sync_read still works there but the
    user can't back-drive the arm to teleop."""
    print("ensuring leader torque is OFF (hand-movable):")
    for sid in IDS:
        _pkt.write1ByteTxRx(_l_ph, sid, ADDR_TORQUE_ENABLE, 0)
    print("  done")


def _apply_safe_limits() -> None:
    """Apply per-motor limits, then enable torque so motors hold present pose."""
    print("applying per-motor settings (upstream so101_follower.configure() values):")
    for sid in IDS:
        joint, torque_cap, accel_cap, p_gain, d_gain = MOTOR_LIMITS[sid]
        # disable torque so EEPROM writes (Max_Torque_Limit, Maximum_Acceleration) take effect
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_TORQUE_ENABLE, 0)
        # reset goal=present so re-enable doesn't lurch
        pos, _, _ = _pkt.read2ByteTxRx(_f_ph, sid, ADDR_PRESENT_POSITION)
        _pkt.write2ByteTxRx(_f_ph, sid, ADDR_GOAL_POSITION, pos)
        # apply per-joint settings
        _pkt.write2ByteTxRx(_f_ph, sid, ADDR_MAX_TORQUE_LIMIT,     torque_cap)
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_MAXIMUM_ACCELERATION, accel_cap)
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_ACCELERATION,         accel_cap)
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_P_COEFFICIENT,        p_gain)
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_D_COEFFICIENT,        d_gain)
        # Gripper-only burnout protection — matches upstream so_follower.py:168-170.
        if sid == 6:
            _pkt.write2ByteTxRx(_f_ph, sid, ADDR_PROTECTION_CURRENT, GRIPPER_PROTECTION_CURRENT)
            _pkt.write1ByteTxRx(_f_ph, sid, ADDR_OVERLOAD_TORQUE,    GRIPPER_OVERLOAD_TORQUE)
        # enable torque (motor holds the goal=present)
        _pkt.write1ByteTxRx(_f_ph, sid, ADDR_TORQUE_ENABLE, 1)
        extra = f"  prot_I={GRIPPER_PROTECTION_CURRENT}  ovld_tq={GRIPPER_OVERLOAD_TORQUE}" if sid == 6 else ""
        print(f"  id={sid} {joint:<14}  torque={torque_cap}  accel={accel_cap}  P={p_gain}  D={d_gain}{extra}")


def _sync_read_retry(ph: PortHandler, group_read: GroupSyncRead) -> dict[int, int]:
    """Run a GroupSyncRead with retries on transient timeout."""
    last = "?"
    for _attempt in range(READ_RETRIES + 1):
        comm = group_read.txRxPacket()
        if comm == 0:
            return {sid: group_read.getData(sid, ADDR_PRESENT_POSITION, 2) for sid in IDS}
        last = f"comm={comm}"
    raise RuntimeError(f"sync_read after {READ_RETRIES + 1} tries: {last}")


def _decode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """Sign-magnitude decode used by STS3215 for Present_Velocity (bit 15)
    and Present_Load (bit 10). Mirrors lerobot.motors.encoding_utils."""
    direction_bit = (value >> sign_bit_index) & 1
    magnitude = value & ((1 << sign_bit_index) - 1)
    return -magnitude if direction_bit else magnitude


def make_follower_pos_vel_load_read() -> GroupSyncRead:
    """Build a GroupSyncRead covering Present_Position, Present_Velocity, and
    Present_Load in a single 6-byte read per motor. One bus round-trip gives
    all three signals — used by data_collector to record velocity/effort
    without inflating the per-tick budget. Requires connect_arms() first."""
    if _f_ph is None or _pkt is None:
        raise RuntimeError("call connect_arms() before make_follower_pos_vel_load_read()")
    g = GroupSyncRead(_f_ph, _pkt, ADDR_PRESENT_POSITION, 6)
    for sid in IDS:
        g.addParam(sid)
    return g


def sync_read_pos_vel_load_retry(group_read: GroupSyncRead) -> dict[int, tuple[int, int, int]]:
    """Retrying read of (position_counts, velocity_counts_per_sec, load_signed)
    per motor. Velocity and load are sign-magnitude-decoded. Returns raw signed
    integers — unit conversion is the caller's responsibility."""
    last = "?"
    for _ in range(READ_RETRIES + 1):
        comm = group_read.txRxPacket()
        if comm == 0:
            out: dict[int, tuple[int, int, int]] = {}
            for sid in IDS:
                pos      = group_read.getData(sid, ADDR_PRESENT_POSITION, 2)
                vel_raw  = group_read.getData(sid, ADDR_PRESENT_VELOCITY, 2)
                load_raw = group_read.getData(sid, ADDR_PRESENT_LOAD,     2)
                out[sid] = (
                    pos,
                    _decode_sign_magnitude(vel_raw,  15),
                    _decode_sign_magnitude(load_raw, 10),
                )
            return out
        last = f"comm={comm}"
    raise RuntimeError(f"sync_read pos+vel+load after {READ_RETRIES + 1} tries: {last}")


def _load_calibration() -> dict[int, tuple[int, int, int, int]]:
    """Return {motor_id: (leader_min, leader_max, follower_min, follower_max)}.

    Used to remap leader raw counts → follower raw counts so the two arms'
    different calibrated ranges line up (leader 0%→follower 0%, 100%→100%).
    Without this, e.g. leader's "fully-closed" gripper raw value is interpreted
    as a partially-open follower goal because the ranges differ.
    """
    for path in (CALIB_LEADER, CALIB_FOLLOWER):
        if not path.exists():
            raise RuntimeError(
                f"Calibration file missing: {path}\n"
                f"Run `lerobot-calibrate` for both arms first."
            )
    with open(CALIB_LEADER)   as f: lcal = json.load(f)
    with open(CALIB_FOLLOWER) as f: fcal = json.load(f)
    out: dict[int, tuple[int, int, int, int]] = {}
    for joint in JOINT_NAMES:
        if joint not in lcal or joint not in fcal:
            raise RuntimeError(f"joint '{joint}' missing from calibration file(s)")
        l = lcal[joint]; f_ = fcal[joint]
        out[l["id"]] = (l["range_min"], l["range_max"],
                        f_["range_min"], f_["range_max"])
    return out


def _remap(leader_raw: int, l_min: int, l_max: int, f_min: int, f_max: int) -> int:
    """Linear range remap: leader's [l_min, l_max] → follower's [f_min, f_max]."""
    if l_max <= l_min:
        return f_min  # degenerate calibration
    frac = (leader_raw - l_min) / (l_max - l_min)
    if frac < 0.0: frac = 0.0
    if frac > 1.0: frac = 1.0
    return int(round(f_min + frac * (f_max - f_min)))


def _sync_write_goals(positions: list[int]) -> None:
    gw = GroupSyncWrite(_f_ph, _pkt, ADDR_GOAL_POSITION, 2)
    for sid, g in zip(IDS, positions):
        g = max(0, min(4095, g))   # clamp to motor range
        gw.addParam(sid, [SCS_LOBYTE(g), SCS_HIBYTE(g)])
    comm = gw.txPacket()
    if comm != 0:
        raise RuntimeError(f"sync_write comm={comm}")


def connect_arms() -> tuple[dict[int, tuple[int, int, int, int]],
                            "GroupSyncRead", "GroupSyncRead"]:
    """Open both ports, load calibration, apply safe limits, build sync_read
    groups. Sets module-level _f_ph / _l_ph / _pkt so module functions
    (_sync_read_retry, _sync_write_goals, _cleanup) work afterwards.

    Returns (calibration, follower_sync_read, leader_sync_read).

    Both this script's main() AND so101_ros2.data_collector call this helper
    so there is exactly one bus path in the package.
    """
    global _f_ph, _l_ph, _pkt
    _f_ph = _open(FOLLOWER_PORT)
    _l_ph = _open(LEADER_PORT)
    _pkt  = PacketHandler(0)

    print("loading calibration…")
    calib = _load_calibration()
    for sid in IDS:
        joint = JOINT_NAMES[sid - 1]
        l_min, l_max, f_min, f_max = calib[sid]
        print(f"  id={sid} {joint:<14}  leader [{l_min},{l_max}]  →  follower [{f_min},{f_max}]")

    _disable_leader_torque()
    _apply_safe_limits()

    # Pre-built sync ops — cheaper than building each iteration
    f_read = GroupSyncRead(_f_ph, _pkt, ADDR_PRESENT_POSITION, 2)
    l_read = GroupSyncRead(_l_ph, _pkt, ADDR_PRESENT_POSITION, 2)
    for sid in IDS:
        f_read.addParam(sid)
        l_read.addParam(sid)
    return calib, f_read, l_read


def main() -> int:
    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"{BOLD}SO-ARM101 safe teleop{RST}")
    print(f"{DIM}follower → {FOLLOWER_PORT}    leader → {LEADER_PORT}{RST}")
    print(f"{DIM}{FPS} Hz, step ≤ {MAX_STEP} counts, retries={READ_RETRIES}{RST}\n")

    calib, f_read, l_read = connect_arms()

    print(f"\n{GREEN}teleop running.{RST} pick up the leader and move slowly. "
          f"{DIM}Ctrl-C to stop.{RST}\n")

    period = 1.0 / FPS
    n_iter = n_fail = 0
    last_status_at = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            try:
                follower_pos = _sync_read_retry(_f_ph, f_read)
                leader_pos   = _sync_read_retry(_l_ph, l_read)
                # Per-joint remap (leader range → follower range), then clip
                # the step so the follower can never slam in one iteration.
                goals = []
                for sid in IDS:
                    l_min, l_max, f_min, f_max = calib[sid]
                    target  = _remap(leader_pos[sid], l_min, l_max, f_min, f_max)
                    present = follower_pos[sid]
                    delta   = target - present
                    if delta >  MAX_STEP: delta =  MAX_STEP
                    if delta < -MAX_STEP: delta = -MAX_STEP
                    goals.append(present + delta)
                _sync_write_goals(goals)
                n_iter += 1
            except Exception as e:
                n_fail += 1
                n_iter += 1
                # Show first few errors, then summarise periodically
                if n_fail <= 3 or n_fail % 30 == 0:
                    print(f"  {YEL}miss{RST} iter {n_iter}: {e}", flush=True)

            now = time.perf_counter()
            if now - last_status_at >= 1.0:
                ok_pct = 100.0 * (n_iter - n_fail) / max(n_iter, 1)
                rate = "OK" if ok_pct >= 99 else "warn" if ok_pct >= 90 else "fail"
                col  = GREEN if ok_pct >= 99 else YEL if ok_pct >= 90 else RED
                print(f"  {col}{rate}{RST}  {n_iter} iter  {ok_pct:5.1f}% success", flush=True)
                last_status_at = now

            rest = period - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
