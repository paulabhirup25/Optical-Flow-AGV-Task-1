import numpy as np
import cv2
import time
from scipy.interpolate import LinearNDInterpolator


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def sparse_to_dense(good_old, flow_vectors, h, w):
    """
    Properly interpolate sparse flow vectors to a dense field.

    Uses LinearNDInterpolator (Delaunay triangulation) so flow vectors
    are correctly spread across the convex hull of tracked points —
    not diluted by zero-filled regions the way a plain blur would do.
    Outside the convex hull fill_value=0 (no motion assumed).
    """
    pts = good_old.reshape(-1, 2)      # (N, 2)  x, y coords
    fx  = flow_vectors[:, 0]
    fy  = flow_vectors[:, 1]

    # All pixel coordinates as query points
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    interp_fx = LinearNDInterpolator(pts, fx, fill_value=0.0)
    interp_fy = LinearNDInterpolator(pts, fy, fill_value=0.0)

    dense_fx = interp_fx(grid_pts).reshape(h, w).astype(np.float32)
    dense_fy = interp_fy(grid_pts).reshape(h, w).astype(np.float32)

    return np.dstack([dense_fx, dense_fy])


# ── Config ────────────────────────────────────────────────────────────────────
SCALE = 0.5      # Process at half resolution for speed

lk_params = dict(
    winSize=(13, 13),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.05)
)

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=5
)
# ─────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture("OPTICAL_FLOW.mp4")

suc, prev = cap.read()
if not suc:
    print("Error: Could not open video")
    exit()

prev     = cv2.resize(prev, None, fx=SCALE, fy=SCALE)
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
p0       = cv2.goodFeaturesToTrack(prevgray, **feature_params)

while True:
    suc, img = cap.read()
    if not suc:
        break

    img  = cv2.resize(img, None, fx=SCALE, fy=SCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    start = time.time()

    flow_dense = np.zeros((h, w, 2), dtype=np.float32)

    if p0 is not None and len(p0) >= 10:

        # Pyramidal LK forward track
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)

        good_new = p1[st.ravel() == 1]
        good_old = p0[st.ravel() == 1]

        if len(good_old) >= 4:
            flow_vectors = (good_new - good_old).reshape(-1, 2)
            # Proper triangulation-based interpolation — arrows will show correctly
            flow_dense   = sparse_to_dense(good_old, flow_vectors, h, w)

        # Re-detect features when count drops
        if len(good_new) < 50:
            p0 = cv2.goodFeaturesToTrack(gray, **feature_params)
        else:
            p0 = good_new.reshape(-1, 1, 2)

    else:
        p0 = cv2.goodFeaturesToTrack(gray, **feature_params)

    prevgray = gray

    end = time.time()
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")

    flow_vis = draw_flow(gray, flow_dense)
    hsv_vis  = draw_hsv(flow_dense)

    cv2.putText(flow_vis, f"{fps:.1f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Dense Flow (Pyramidal LK)", flow_vis)
    cv2.imshow("Dense Flow HSV", hsv_vis)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()