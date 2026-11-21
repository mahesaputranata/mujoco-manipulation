"""Figure for the README: what the fixed camera sees, and what perception.py extracts from it.

    python plot_perception.py   ->  results/perception/camera_view.png

Left: the camera image. Right: the red blob (dimmed), the top face actually used (red), the
estimated cube outline projected back into the image (green) and the true outline (white).
"""
import cv2
import numpy as np

from eval_place import ROOT
from perception import CameraPerception, PinholeCamera, estimate_from_image, largest_blob, red_mask, top_face
from scene import build_model, place_cube

GREEN, WHITE = (60, 220, 90), (255, 255, 255)


def square(x, y, yaw, half, z):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.c_[pts, np.full(4, z)]


def draw_outline(img, camera, x, y, yaw, half, color, thickness):
    """Top and bottom squares of a cube and its vertical edges, projected into the image."""
    top = camera.project(square(x, y, yaw, half, 2 * half))
    bottom = camera.project(square(x, y, yaw, half, 0.0))
    for poly in (top, bottom):
        cv2.polylines(img, [np.round(poly * 4).astype(np.int32)], True, color, thickness, cv2.LINE_AA, shift=2)
    for a, b in zip(top, bottom):
        cv2.line(img, tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int)), color, thickness,
                 cv2.LINE_AA, shift=2)


def main(x=0.48, y=0.22, yaw_deg=25.0, half=0.025, crop=(314, 211, 80, 60), S=6):
    model, data = build_model(cube_half=half)
    place_cube(model, data, (x, y), np.radians(yaw_deg))
    cam = CameraPerception(model)
    rgb = cam.image(data).copy()
    cam.close()
    camera = PinholeCamera(model, data)
    est = estimate_from_image(rgb, camera)

    blob = largest_blob(red_mask(rgb))
    face = top_face(rgb, blob)
    overlay = (rgb * 0.45).astype(np.uint8)
    overlay[blob] = (120, 40, 40)
    overlay[face] = (235, 50, 50)
    draw_outline(overlay, camera, x, y, np.radians(yaw_deg), half, WHITE, 3)  # truth: thick, underneath
    draw_outline(overlay, camera, est["pos"][0], est["pos"][1], est["yaw"], est["half"], GREEN, 1)

    # Zoom both panels on the cube so the outlines are visible, then put them side by side.
    u0, v0, w, h = crop
    zoom = lambda im: cv2.resize(im[v0:v0 + h, u0:u0 + w], (w * S, h * S), interpolation=cv2.INTER_NEAREST)
    left = cv2.resize(rgb, (int(rgb.shape[1] * h * S / rgb.shape[0]), h * S), interpolation=cv2.INTER_AREA)
    cv2.rectangle(left, (int(u0 * left.shape[1] / rgb.shape[1]), int(v0 * left.shape[0] / rgb.shape[0])),
                  (int((u0 + w) * left.shape[1] / rgb.shape[1]), int((v0 + h) * left.shape[0] / rgb.shape[0])),
                  WHITE, 2)
    panel = np.hstack([left, np.full((h * S, 12, 3), 255, np.uint8), zoom(overlay)])

    err = 1000 * np.linalg.norm(est["pos"][:2] - [x, y])
    d_yaw = np.degrees((est["yaw"] - np.radians(yaw_deg) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    bar = np.full((60, panel.shape[1], 3), 252, np.uint8)
    cv2.putText(bar, "Fixed RGB camera (left). Zoom (right): red blob, top face used, estimate (green) on truth (white)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(bar, f"Estimate vs truth: position {err:.1f} mm, yaw {d_yaw:+.1f} deg, "
                     f"size {1000 * 2 * est['half']:.1f} mm (true {1000 * 2 * half:.0f} mm)",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    out = ROOT / "results" / "perception"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "camera_view.png"), cv2.cvtColor(np.vstack([panel, bar]), cv2.COLOR_RGB2BGR))
    print(f"saved {out / 'camera_view.png'}  (position error {err:.2f} mm, yaw error {d_yaw:+.2f} deg)")


if __name__ == "__main__":
    main()
