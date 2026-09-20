"""
Axis Diagnostic Tool
====================
Run this script, press 'o' to lock the origin, then move the marker
in ONE direction at a time. It will tell you which robot axis responds.
"""
import cv2
import numpy as np
from webcam_aruco_tracker import WebcamArucoTracker

tracker = WebcamArucoTracker(marker_length=0.093, alpha=0.65, camera_index=1)
tracker.start()

origin = None
print("Hold marker in front of camera. Press 'o' to set origin.")
print("Then move the marker in ONE direction at a time and observe.\n")

while True:
    positions, frame = tracker.step()
    if frame is None:
        continue

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):
        if positions and positions.get(0) is not None:
            origin = positions[0].copy()
            print(">>> ORIGIN SET <<<\n")
            print("Now slowly move the marker in these directions:")
            print("  1) LEFT / RIGHT  (your left/right as you face the camera)")
            print("  2) UP / DOWN     (lift or lower the marker)")
            print("  3) FORWARD / BACKWARD (towards/away from camera)\n")

    if origin is not None and positions and positions.get(0) is not None:
        pose = positions[0]
        dx = pose[0] - origin[0]
        dy = pose[1] - origin[1]
        dz = pose[2] - origin[2]
        drx = pose[3] - origin[3]
        dry = pose[4] - origin[4]
        drz = pose[5] - origin[5]

        # Find the dominant axis
        deltas = {'X': dx, 'Y': dy, 'Z': dz}
        dominant = max(deltas, key=lambda k: abs(deltas[k]))
        dominant_val = deltas[dominant]

        bar_len = int(min(abs(dominant_val) * 500, 40))
        direction = "+" if dominant_val > 0 else "-"
        bar = direction * bar_len

        print(f"  dX={dx:+.4f}  dY={dy:+.4f}  dZ={dz:+.4f}  |  Dominant: {dominant}{direction} [{bar:<40s}]  |  dRX={drx:+.2f} dRY={dry:+.2f} dRZ={drz:+.2f}", end="\r")

    if frame is not None:
        if origin is not None:
            cv2.putText(frame, "ORIGIN SET - Move marker slowly", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press 'o' to set origin", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.imshow("Axis Diagnostic", frame)

tracker.stop()
cv2.destroyAllWindows()

