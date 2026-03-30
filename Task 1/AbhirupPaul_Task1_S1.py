import numpy as np
import cv2
import time

# Pyramidal LK params — tuned for speed:
# - maxLevel=3 gives 4 pyramid levels (coarser search handles large motion faster)
# - Reduced iterations (7 vs 10) and looser epsilon (0.05 vs 0.03)
# - Smaller winSize (11x11) reduces the per-pixel search area
lk_params = dict(
    winSize=(11, 11),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.05)
)

feature_params = dict(
    maxCorners=20,
    qualityLevel=0.3,
    minDistance=10,
    blockSize=7
)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture("OPTICAL_FLOW.mp4")

# Resize scale factor — process at half resolution, display at full
SCALE = 0.5

mask = None
prev_gray = None

while True:
    start = time.time()

    suc, frame = cap.read()
    if not suc:
        break

    # Downscale for processing
    small_frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    frame_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    img = small_frame.copy()

    if prev_gray is None:
        prev_gray = frame_gray
        continue

    # --- Optical Flow Tracking ---
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray

        p0 = np.float32([t[-1] for t in trajectories]).reshape(-1, 1, 2)

        # Forward pass
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Backward pass for consistency check
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(t) for t in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories),
                    (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # --- Detect new features every few frames ---
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(t[-1]) for t in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(img, f"{fps:.2f} FPS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Sparse Optical Flow (Fast)", img)
    if mask is not None:
        cv2.imshow("Mask", mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()