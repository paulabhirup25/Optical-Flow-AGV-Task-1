"""
vpf_navigation.py
=================
Based on the reference VPF code, with the following targeted fixes:

FIX 1 — Ray direction sign (raycast_obstacle):
  The reference uses `right = rot col-1` and `d = cos*fwd + sin*right`.
  Col-1 of the rotation matrix is the car's LEFT vector (body-y = world +y
  when heading +x).  So positive angles fanned to the LEFT, but were labelled
  as left_dists — which happens to be correct for labelling, but then the
  hit_pos world-y is used directly, so the sign doesn't matter there.
  The real fix is just using world-y of the hit point for direction (Fix 3).

FIX 2 — Dodge direction uses world-y of closest hit point:
  Instead of comparing dist_left vs dist_right (biased by car's lateral
  position), we read the world-space Y coordinate of the closest raycast hit.
  obstacle world-y > 0 → obstacle is LEFT  → dodge RIGHT (negative target_y)
  obstacle world-y < 0 → obstacle is RIGHT → dodge LEFT  (positive target_y)
  This is an absolute measurement, completely independent of where the car is.

FIX 3 — Per-obstacle tracking (dodge locked per obstacle x-position):
  Each obstacle is identified by its approximate world-x position.
  Once a dodge direction is chosen for an obstacle, it is locked until the
  car has passed that obstacle (car_x > obs_x + 1.5 m).
  This prevents the same obstacle from being re-evaluated on every frame,
  and ensures each new obstacle gets a fresh direction choice.

FIX 4 — Dodge force is a fixed step function, not a distance-scaled ramp:
  The reference scales dodge by urgency (0→1).  A partial dodge does not
  clear the obstacle.  When an obstacle is detected, the car moves to a
  fixed target_y = ±DODGE_Y_TARGET regardless of distance.

FIX 5 — GTSMC integrator reset on new dodge:
  When a new obstacle triggers a dodge, delta_f is reset to 0 so there is
  no wind-up from the previous manoeuvre.

FIX 6 — Cooldown cannot block a new obstacle lock:
  If a new threat arrives while in cooldown, cooldown is cancelled immediately
  and the new obstacle is locked. This prevents obstacle 4 (or any obstacle
  arriving during re-centring from obstacle 3) from being silently skipped.

FIX 7 — obs_x estimated from raycast distance, not a fixed +4.0 m offset:
  _dodge_obs_x = car_x + min_dist gives a much more accurate world-x for the
  obstacle, so the release check (car_x > obs_x + margin) fires at the right
  time instead of too early.

FIX 8 — delta_f bleed on every new dodge:
  When _dodge_active is set for a new obstacle, delta_f is explicitly reset
  to 0 to prevent steering wind-up from previous manoeuvres carrying over.

Everything else (GTSMC structure, centring force PD, hysteresis, boundary
guard, optical flow visualisation, HUD) is kept exactly from the reference.
"""

import cv2
import numpy as np
import pybullet as p
import time
from simulation_setup import setup_simulation

# ── Camera ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FOV           = 90

# ── Road geometry ─────────────────────────────────────────────────────────────
ROAD_HALF_W = 1.16
CAR_HALF_W  = 0.25

# ── Cruising ──────────────────────────────────────────────────────────────────
SPEED_REF = 10.0
STEER_MAX = 0.6

# ── Centring controller ───────────────────────────────────────────────────────
CENTRE_KP = 8.0
CENTRE_KD = 2.0

# ── Raycast threat thresholds ─────────────────────────────────────────────────
DIST_WARN  = 12.0
DIST_BRAKE =  7.0
DIST_EMERG =  3.5

OBS_SPEED_WARN      = 6.0
OBS_SPEED_BRAKE     = 3.5
OBS_SPEED_EMERGENCY = 1.5

# Ray fan
RAY_ANGLES_DEG = [-25, -10, 0, 10, 25]
RAY_LENGTH     = 15.0
RAY_HEIGHT     = 0.35

# ── Threat hysteresis ─────────────────────────────────────────────────────────
THREAT_DEESCALATE_FRAMES = 6

# ── Dodge parameters ──────────────────────────────────────────────────────────
DODGE_Y_TARGET = 0.55

# FIX 7: release margin is now measured from the obstacle's true estimated x.
# Increased to 2.5 m so the car is safely past before releasing.
DODGE_RELEASE_MARGIN = 2.5   # [m]

# ── Post-dodge re-centring ────────────────────────────────────────────────────
RECENTER_COOLDOWN = 1.0      # reduced: shorter cooldown window so obs 4 can lock
RECENTER_BOOST    = 4.0
RECENTER_FY_DECAY = 0.20     # faster decay so cooldown clears before next obs

# ── Hard boundary guard ───────────────────────────────────────────────────────
BOUNDARY_Y     = 0.72
BOUNDARY_STEER = 0.6
BOUNDARY_SPEED = 3.0

# ── VPF / APF parameters (unchanged from reference) ──────────────────────────
APF_A           = 0.5
APF_B           = 1.0
APF_C2_STRAIGHT = 0.005
APF_C2_CURVED   = 5e-6
APF_C1          = 0.0
APF_C0R         =  0.91
APF_C0L         = -0.91
DELTA_X         = 1.0e-10
ALPHA_ATT       = 0.5
GAMMA_REP       = 1.5
LAMBDA_X        = 0.4
LAMBDA_Y        = 0.4
SIGMA_GAUSS     = WIDTH / 2.0

# ── GTSMC ─────────────────────────────────────────────────────────────────────
CR = 2.0
U0 = 0.6
CL = 1.5
A0 = 3.0

# ── Optical flow ──────────────────────────────────────────────────────────────
lk_params = dict(winSize=(25, 25), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                            cv2.TERM_CRITERIA_COUNT, 20, 0.03))
feature_params = dict(maxCorners=300, qualityLevel=0.01,
                      minDistance=8, blockSize=7)

# ── Module state ──────────────────────────────────────────────────────────────
prev_gray  = None
prev_pts   = None
delta_f    = 0.0
speed_cur  = SPEED_REF

_prev_y          = 0.0
_smooth_dodge_FY = 0.0
_cooldown_timer  = 0.0
_last_threat     = "clear"

_candidate_threat = "clear"
_candidate_frames = 0
_confirmed_threat = "clear"

# Per-obstacle dodge lock (FIX 3 + FIX 7)
_dodge_obs_x      = -999.0
_dodge_target_y   =  0.0
_dodge_active     = False

# Track the last min_dist seen so compute_dodge_force can use it (FIX 7)
_last_min_dist    = RAY_LENGTH


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

def get_camera_image(car_id):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot      = p.getMatrixFromQuaternion(orn)
    forward  = np.array([rot[0], rot[3], rot[6]])
    cam_pos  = np.array(pos) + forward * 1.05 + np.array([0, 0, 0.32])
    target   = cam_pos + forward * 5 + np.array([0, 0, -0.25])
    view     = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
    proj     = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, 0.1, 50)
    _, _, rgb, _, _ = p.getCameraImage(WIDTH, HEIGHT, view, proj)
    frame_rgb  = np.array(rgb, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))[:, :, :3]
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return frame_gray, frame_rgb


# ═══════════════════════════════════════════════════════════════════════════════
# RAYCASTING
# ═══════════════════════════════════════════════════════════════════════════════

def raycast_obstacle(car_id):
    """
    Cast five rays ahead.  Returns:
      min_dist      : closest hit [m]
      hit_world_y   : world-Y of the closest hit point
      dist_left     : mean distance of left-angle rays
      dist_right    : mean distance of right-angle rays
      hits          : list of (dist, ang_deg, did_hit) for visualisation
    """
    pos, orn  = p.getBasePositionAndOrientation(car_id)
    rot       = p.getMatrixFromQuaternion(orn)
    fwd       = np.array([rot[0], rot[3], rot[6]])
    left_vec  = np.array([rot[1], rot[4], rot[7]])

    ray_start = np.array(pos) + fwd * 0.6 + np.array([0, 0, RAY_HEIGHT])

    hits       = []
    best_dist  = RAY_LENGTH
    best_hit_y = float(pos[1])

    for ang_deg in RAY_ANGLES_DEG:
        ang = np.radians(ang_deg)
        d   = np.cos(ang) * fwd + np.sin(ang) * left_vec
        end = ray_start + d * RAY_LENGTH
        res = p.rayTest(ray_start.tolist(), end.tolist())[0]
        hid = res[0]
        if hid > 0 and hid != car_id:
            hit_pos  = np.array(res[3])
            hit_dist = float(np.linalg.norm(hit_pos - ray_start))
            if hit_dist < best_dist:
                best_dist  = hit_dist
                best_hit_y = float(hit_pos[1])
        else:
            hit_dist = RAY_LENGTH
        hits.append((hit_dist, ang_deg, hid > 0 and hid != car_id))

    all_dists  = [h[0] for h in hits]
    min_dist   = min(all_dists)
    left_dists  = [h[0] for h in hits if h[1] > 0]
    right_dists = [h[0] for h in hits if h[1] < 0]
    centre_d    = next((h[0] for h in hits if h[1] == 0), RAY_LENGTH)

    dist_left  = float(np.mean(left_dists))  if left_dists  else RAY_LENGTH
    dist_right = float(np.mean(right_dists)) if right_dists else RAY_LENGTH
    if dist_left < dist_right:
        dist_left  = min(dist_left,  centre_d)
    else:
        dist_right = min(dist_right, centre_d)

    return min_dist, best_hit_y, dist_left, dist_right, hits


def classify_raw_threat(dist):
    if dist > DIST_WARN:  return "clear"
    elif dist > DIST_BRAKE: return "warn"
    elif dist > DIST_EMERG: return "brake"
    else:                   return "emergency"


THREAT_RANK = {"clear": 0, "warn": 1, "brake": 2, "emergency": 3}

def apply_hysteresis(raw_threat):
    global _candidate_threat, _candidate_frames, _confirmed_threat
    raw_rank  = THREAT_RANK.get(raw_threat, 0)
    conf_rank = THREAT_RANK.get(_confirmed_threat, 0)
    if raw_rank >= conf_rank:
        _confirmed_threat = raw_threat
        _candidate_threat = raw_threat
        _candidate_frames = THREAT_DEESCALATE_FRAMES
    else:
        if raw_threat == _candidate_threat:
            _candidate_frames += 1
        else:
            _candidate_threat = raw_threat
            _candidate_frames = 1
        if _candidate_frames >= THREAT_DEESCALATE_FRAMES:
            _confirmed_threat = _candidate_threat
    return _confirmed_threat


# ═══════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW + FOE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_optical_flow(prev_g, curr_g, prev_p):
    if prev_p is None or len(prev_p) < 30:
        prev_p = cv2.goodFeaturesToTrack(prev_g, mask=None, **feature_params)
        if prev_p is None:
            return None, None, None, None
    next_p, status, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, prev_p, None, **lk_params)
    if next_p is None or status is None:
        return None, None, None, None
    mask     = status.ravel() == 1
    good_old = prev_p[mask];  good_new = next_p[mask]
    if len(good_new) == 0:
        return None, None, None, None
    return good_old, good_new, good_new - good_old, good_new.reshape(-1, 1, 2)


def compute_foe(good_old, flow_vecs):
    if good_old is None or len(good_old) < 4:
        return WIDTH / 2.0, HEIGHT / 2.0
    pts = good_old.reshape(-1, 2);  vxy = flow_vecs.reshape(-1, 2)
    A   = np.column_stack([vxy[:, 1], -vxy[:, 0]])
    b   = pts[:, 0] * vxy[:, 1] - pts[:, 1] * vxy[:, 0]
    try:
        ATA = A.T @ A
        det = ATA[0,0]*ATA[1,1] - ATA[0,1]**2
        if abs(det) < 1e-8: return WIDTH/2.0, HEIGHT/2.0
        foe = np.linalg.solve(ATA, A.T @ b)
        return float(np.clip(foe[0], 0, WIDTH)), float(np.clip(foe[1], 0, HEIGHT))
    except np.linalg.LinAlgError:
        return WIDTH / 2.0, HEIGHT / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRING FORCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_centre_force(car_id, dt, boost=1.0):
    global _prev_y
    pos, _  = p.getBasePositionAndOrientation(car_id)
    y       = float(pos[1])
    dy_dt   = (y - _prev_y) / max(dt, 1e-4)
    _prev_y = y
    return float(-(y * CENTRE_KP + dy_dt * CENTRE_KD) * boost), abs(y)


# ═══════════════════════════════════════════════════════════════════════════════
# DODGE FORCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dodge_force(threat, hit_world_y, car_x, car_y, min_dist, dt):
    """
    FIX 2: direction from hit_world_y.
    FIX 3: per-obstacle lock — held until car passes obs_x + margin.
    FIX 4: fixed target_y = ±DODGE_Y_TARGET.
    FIX 6: new threat cancels cooldown immediately so obstacle 4 is never skipped.
    FIX 7: obs_x estimated as car_x + min_dist (not a fixed +4.0 offset).
    FIX 8: delta_f reset to 0 on every new dodge lock.
    """
    global _smooth_dodge_FY, _cooldown_timer, _last_threat
    global _dodge_obs_x, _dodge_target_y, _dodge_active
    global delta_f

    # Speed target
    if threat == "clear":
        obs_speed = SPEED_REF
    elif threat == "warn":
        obs_speed = OBS_SPEED_WARN
    elif threat == "brake":
        obs_speed = OBS_SPEED_BRAKE
    else:
        obs_speed = OBS_SPEED_EMERGENCY

    # Cooldown transition
    if _last_threat != "clear" and threat == "clear":
        _cooldown_timer = RECENTER_COOLDOWN
    elif threat != "clear":
        _cooldown_timer = 0.0
    _last_threat = threat

    in_cooldown = (_cooldown_timer > 0.0) and (threat == "clear")
    if in_cooldown:
        _cooldown_timer = max(0.0, _cooldown_timer - dt)

    # FIX 6 — A new threat cancels cooldown immediately.
    # Without this, obstacle 4 arriving during the cooldown from obstacle 3
    # is silently skipped because the lock branch requires `not in_cooldown`.
    if threat != "clear" and in_cooldown:
        _cooldown_timer  = 0.0
        in_cooldown      = False
        _smooth_dodge_FY = 0.0   # discard residual decay from previous dodge

    # FIX 3: release lock once car has passed the obstacle
    if _dodge_active and car_x > _dodge_obs_x + DODGE_RELEASE_MARGIN:
        _dodge_active   = False
        _dodge_obs_x    = -999.0
        _dodge_target_y = 0.0
        _cooldown_timer = RECENTER_COOLDOWN

    # FIX 3 + 7: lock onto a new obstacle
    if threat != "clear" and not _dodge_active and not in_cooldown:
        if abs(hit_world_y) < 0.08:
            _dodge_target_y = -DODGE_Y_TARGET
        elif hit_world_y > 0:
            _dodge_target_y = -DODGE_Y_TARGET   # obstacle left → go right
        else:
            _dodge_target_y =  DODGE_Y_TARGET   # obstacle right → go left

        # FIX 7: use actual raycast distance for obs_x estimate
        _dodge_obs_x  = car_x + min_dist
        _dodge_active = True

        # FIX 8: clear integrator wind-up from previous dodge
        delta_f = 0.0

        print(f"[DODGE] hit_y={hit_world_y:+.2f} → target_y={_dodge_target_y:+.2f}"
              f"  car_y={car_y:+.2f}  obs_x≈{_dodge_obs_x:.1f}"
              f"  dist={min_dist:.1f}m")

    # Compute dodge_FY
    if _dodge_active and not in_cooldown:
        dodge_FY = 12.0 * (_dodge_target_y - car_y)
        _smooth_dodge_FY = dodge_FY
    elif in_cooldown:
        _smooth_dodge_FY *= (1.0 - RECENTER_FY_DECAY)
        if abs(_smooth_dodge_FY) < 0.01:
            _smooth_dodge_FY = 0.0
    else:
        _smooth_dodge_FY = 0.0

    centre_boost     = RECENTER_BOOST if in_cooldown else 1.0
    bleed_integrator = in_cooldown
    return float(_smooth_dodge_FY), float(obs_speed), in_cooldown, centre_boost, bleed_integrator


# ═══════════════════════════════════════════════════════════════════════════════
# VPF FORCES  (unchanged from reference)
# ═══════════════════════════════════════════════════════════════════════════════

def build_obstacle_map(good_new, flow_vecs, xFOE, yFOE):
    O = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if good_new is not None and len(good_new) > 0:
        pts = good_new.reshape(-1, 2);  vxy = flow_vecs.reshape(-1, 2)
        df  = np.sqrt((pts[:,0]-xFOE)**2 + (pts[:,1]-yFOE)**2) + 1e-6
        fm  = np.sqrt(vxy[:,0]**2 + vxy[:,1]**2) + 1e-6
        dot = vxy[:,0]*(pts[:,0]-xFOE)/df + vxy[:,1]*(pts[:,1]-yFOE)/df
        res = fm - np.abs(dot)
        xs  = np.clip(pts[:,0].astype(int), 0, WIDTH-1)
        ys  = np.clip(pts[:,1].astype(int), 0, HEIGHT-1)
        for i in range(len(xs)):
            O[ys[i], xs[i]] = max(O[ys[i], xs[i]], float(res[i]))
    O_u8 = cv2.normalize(O, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if O_u8.max() > 0:
        _, mask = cv2.threshold(O_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = O_u8
    k  = min(int(6*SIGMA_GAUSS)|1, min(WIDTH,HEIGHT)-1)|1
    bl = cv2.GaussianBlur(mask.astype(np.float32), (k, k), SIGMA_GAUSS)
    return mask, cv2.Sobel(bl, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(bl, cv2.CV_32F, 0, 1, ksize=3)


def compute_road_potential_field(car_id, xFOE):
    pos, _ = p.getBasePositionAndOrientation(car_id)
    x_veh  = float(pos[0]);  y_veh = float(pos[1])
    foe_n  = xFOE / WIDTH
    c2     = APF_C2_STRAIGHT if 0.35 < foe_n < 0.65 else (-APF_C2_CURVED if foe_n <= 0.35 else APF_C2_CURVED)
    xd     = x_veh + DELTA_X
    yr     = c2*xd**2 + APF_C0R;  yl = c2*xd**2 + APF_C0L
    denom  = 2*c2*xd + APF_C1;    m  = -1.0 / (denom if abs(denom) > 1e-9 else 1e-9)
    by_r   = yr - m*xd;            by_l = yl - m*xd
    eps    = 1e-4

    def Utotal(y_):
        yr_ = c2*xd**2 + APF_C0R;  yl_ = c2*xd**2 + APF_C0L
        def _U(yb, by_, s):
            inner = np.sqrt(((y_-by_)/m - xd)**2 + (yb-y_)**2 + 1e-9)
            return APF_A*(1 - np.exp(-np.clip(s*APF_B*inner,-50,50)))**2
        sr = np.sign(y_-yr_) if abs(y_-yr_)>1e-6 else 1.0
        sl = np.sign(y_-yl_) if abs(y_-yl_)>1e-6 else -1.0
        return _U(yr_, by_r, sr) + _U(yl_, by_l, sl)

    return 0.0, float(-(Utotal(y_veh+eps) - Utotal(y_veh-eps)) / (2*eps))


def compute_obstacle_force(good_new, flow_vecs, ttc_arr, grad_x, grad_y):
    if good_new is None or len(good_new) == 0:
        return 0.0, 0.0
    pts = good_new.reshape(-1, 2).astype(int)
    xs  = np.clip(pts[:,0], 0, WIDTH-1);  ys = np.clip(pts[:,1], 0, HEIGHT-1)
    tv  = ttc_arr if len(ttc_arr)==len(xs) else np.ones(len(xs))
    ts  = max(float(tv.sum()), 1e-6);  n = max(len(xs), 1)
    return (float(GAMMA_REP/n * grad_x[ys,xs].sum() / ts),
            float(GAMMA_REP/n * grad_y[ys,xs].sum() / ts))


def compute_ttc(good_new, flow_vecs, xFOE, yFOE):
    if good_new is None or len(good_new) == 0:
        return np.array([])
    pts = good_new.reshape(-1, 2);  vxy = flow_vecs.reshape(-1, 2)
    return np.sqrt((pts[:,0]-xFOE)**2 + (pts[:,1]-yFOE)**2) / \
           (np.sqrt(vxy[:,0]**2 + vxy[:,1]**2) + 1e-6)


def compute_total_force(car_id, xFOE, good_new, flow_vecs, ttc_arr,
                        grad_x, grad_y, yaw, centre_FY, dodge_FY):
    pos, _  = p.getBasePositionAndOrientation(car_id)
    Fatt_x  = ALPHA_ATT * 1.0
    Fatt_y  = ALPHA_ATT * (-float(pos[1]))
    Fo_x, Fo_y = compute_obstacle_force(good_new, flow_vecs, ttc_arr, grad_x, grad_y)
    Fr_x, Fr_y = compute_road_potential_field(car_id, xFOE)
    FXT = Fatt_x - Fo_x - LAMBDA_X * Fr_x
    FYT = Fatt_y - Fo_y - LAMBDA_Y * Fr_y + centre_FY + dodge_FY
    c = np.cos(yaw);  s = np.sin(yaw)
    return float(c*FXT + s*FYT), float(-s*FXT + c*FYT)


# ═══════════════════════════════════════════════════════════════════════════════
# GTSMC
# ═══════════════════════════════════════════════════════════════════════════════

def gtsmc_lateral(car_id, FX0, FY0, dt):
    global delta_f
    psi_d   = float(np.arctan2(FY0, FX0 + 1e-9))
    _, orn  = p.getBasePositionAndOrientation(car_id)
    psi     = float(p.getEulerFromQuaternion(orn)[2])
    psi_e   = (psi - psi_d + np.pi) % (2*np.pi) - np.pi
    sr      = CR * psi_e + psi_e / max(dt, 1e-4)
    delta_f = float(np.clip(delta_f + (-U0 * float(np.sign(sr))) * dt, -STEER_MAX, STEER_MAX))
    return delta_f


def gtsmc_longitudinal(obs_speed, dt):
    global speed_cur
    sl        = CL * speed_cur - obs_speed
    speed_cur = float(np.clip(speed_cur + (-A0 * float(np.sign(sl))) * dt, 0.5, SPEED_REF + 1.0))
    return speed_cur


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global prev_gray, prev_pts, delta_f, speed_cur

    car_id, steer_j, motor_j = setup_simulation()
    print("[VPF] Starting")
    last_t = time.time()

    while True:
        p.stepSimulation()
        now = time.time();  dt = max(now - last_t, 1e-4);  last_t = now

        gray, rgb = get_camera_image(car_id)

        if prev_gray is None:
            prev_gray = gray
            prev_pts  = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            continue

        good_old, good_new, flow_vecs, prev_pts = compute_optical_flow(prev_gray, gray, prev_pts)
        prev_gray = gray

        xFOE, yFOE = compute_foe(good_old, flow_vecs)
        ttc_arr    = compute_ttc(good_new, flow_vecs, xFOE, yFOE)
        obs_mask, grad_x, grad_y = build_obstacle_map(good_new, flow_vecs, xFOE, yFOE)

        pos_now, orn = p.getBasePositionAndOrientation(car_id)
        car_x  = float(pos_now[0])
        car_y  = float(pos_now[1])
        yaw    = float(p.getEulerFromQuaternion(orn)[2])

        # ── Raycasts ──────────────────────────────────────────────────────────
        min_dist, hit_world_y, dist_left, dist_right, ray_hits = raycast_obstacle(car_id)
        raw_threat       = classify_raw_threat(min_dist)
        confirmed_threat = apply_hysteresis(raw_threat)

        # ── Dodge force (FIX 2, 3, 4, 6, 7, 8) ───────────────────────────────
        dodge_FY, obs_speed, in_cooldown, centre_boost, bleed_integrator = \
            compute_dodge_force(confirmed_threat, hit_world_y, car_x, car_y, min_dist, dt)

        # ── Centring force ─────────────────────────────────────────────────────
        centre_FY, abs_y = compute_centre_force(car_id, dt, boost=centre_boost)

        # ── GTSMC integrator bleed ────────────────────────────────────────────
        if bleed_integrator and abs_y > 0.05:
            delta_f *= 0.92

        # ── Total force ───────────────────────────────────────────────────────
        FX0, FY0 = compute_total_force(car_id, xFOE, good_new, flow_vecs, ttc_arr,
                                       grad_x, grad_y, yaw, centre_FY, dodge_FY)

        # ── GTSMC ─────────────────────────────────────────────────────────────
        steering = gtsmc_lateral(car_id, FX0, FY0, dt)
        speed    = gtsmc_longitudinal(obs_speed, dt)

        # ── Emergency override ────────────────────────────────────────────────
        if confirmed_threat == "emergency":
            if abs(hit_world_y) < 0.08:
                dodge_dir = 1.0 if dist_left >= dist_right else -1.0
            else:
                dodge_dir = -1.0 if hit_world_y > 0 else 1.0
            steering = dodge_dir * STEER_MAX
            delta_f  = steering
            speed    = max(speed, OBS_SPEED_BRAKE)

        # ── Hard boundary guard ───────────────────────────────────────────────
        boundary_active = abs(car_y) > BOUNDARY_Y
        if boundary_active:
            steering = -float(np.sign(car_y)) * BOUNDARY_STEER
            delta_f  = steering
            speed    = min(speed, BOUNDARY_SPEED)

        # ── Actuation ─────────────────────────────────────────────────────────
        for j in steer_j:
            p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL,
                                    targetPosition=steering, force=250, positionGain=0.3)
        for j in motor_j:
            p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL,
                                    targetVelocity=speed, force=1200)

        # ── Visualisation ─────────────────────────────────────────────────────
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if good_new is not None and good_old is not None:
            for n_pt, o_pt in zip(good_new.reshape(-1,2), good_old.reshape(-1,2)):
                cv2.arrowedLine(vis, (int(o_pt[0]),int(o_pt[1])),
                                (int(n_pt[0]),int(n_pt[1])), (0,200,0), 1, tipLength=0.4)

        cv2.circle(vis, (int(xFOE), int(yFOE)), 6, (0,0,255), -1)

        threat_col = {"clear":(0,255,0),"warn":(0,255,255),
                      "brake":(0,165,255),"emergency":(0,0,255)}
        col = threat_col.get(confirmed_threat, (0,255,0))

        for (hit_dist, ang_deg, did_hit) in ray_hits:
            bar_x = int(WIDTH/2 + ang_deg * (WIDTH / (2*max(RAY_ANGLES_DEG))))
            bar_h = int((1.0 - hit_dist / RAY_LENGTH) * 60)
            cv2.rectangle(vis, (bar_x-6, HEIGHT-10-bar_h), (bar_x+6, HEIGHT-10),
                          (0,0,255) if did_hit else (0,200,0), -1)

        cv2.line(vis, (WIDTH//2, 0), (WIDTH//2, HEIGHT), (255,255,255), 1)

        mo = np.zeros_like(vis);  mo[:,:,2] = obs_mask
        vis = cv2.addWeighted(vis, 1.0, mo, 0.3, 0)

        status = "BOUNDARY!" if boundary_active else \
                 ("DODGE"    if _dodge_active else \
                 ("RECENTRE" if in_cooldown else "CENTRE"))
        hud = [
            f"Mode:    {status}",
            f"Threat:  {confirmed_threat.upper()}  (raw:{raw_threat})",
            f"Dist:    {min_dist:.1f}m  hit_y:{hit_world_y:+.2f}  L:{dist_left:.1f}  R:{dist_right:.1f}",
            f"Speed:   {speed:.1f} m/s",
            f"Steer:   {steering:+.3f} rad",
            f"Car Y:   {car_y:+.3f} m   target:{_dodge_target_y:+.2f}",
        ]
        for i, txt in enumerate(hud):
            c = col if i == 1 else (0,0,220) if boundary_active and i==0 else (0,255,180)
            cv2.putText(vis, txt, (8, 22+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 1)

        cv2.imshow("VPF Navigation", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    p.disconnect()
    print("[VPF] Done.")


if __name__ == "__main__":
    main()