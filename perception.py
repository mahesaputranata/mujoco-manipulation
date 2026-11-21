"""Camera-based perception: find the red cube in an RGB image and estimate its pose on the table.

Replaces the "oracle" (the simulator telling the planner where the cube is) with what a fixed
RGB camera can see. Pipeline:

    render RGB  ->  colour mask  ->  top face  ->  rays through its outline  ->  intersect with the
    plane of the cube's top  ->  rectangle fit  ->  centre (x, y), yaw, size  ->  cube pose

Geometry, not the detector, is the hard part. A camera looking down at 45 deg also sees the cube's
sides, so the centre of the red blob is not the centre of the cube. Only the top face is used: it
lies in a known plane (z = cube height), so its outline can be mapped to the table exactly.

The cube's height is not known in advance (it varies with --rand-physics). A cube is as tall as
it is wide, so the estimate is iterated: assume a height, map the top face to that plane, measure
its width, use that as the new height. This converges in a few steps.
"""
import cv2
import mujoco
import numpy as np

from scene import CUBE_HALF, SCENE_CAMERA

WIDTH, HEIGHT = 640, 480
MIN_PIXELS = 30  # a smaller red blob is treated as "cube not seen" (occluded or out of view)
HALF_RANGE = (0.012, 0.040)  # plausible cube half-sizes (m); outside this the estimate is rejected
AGREE_POS, AGREE_YAW = 0.002, np.radians(2.0)  # two looks must agree this closely
MAX_LOOKS = 5
# How far (pixels) the true edge lies outside the centres of the outermost mask pixels. Half a
# pixel would be right for hard-edged rasterization, but MuJoCo anti-aliases edges, so partly
# covered pixels are already in the mask. Measured on 40 rendered cubes: 0.5 px overestimates
# the size by 0.6 mm and gives 1.3 mm position error; 0.25 px gives +0.1 mm and 0.6 mm.
EDGE_PUSH = 0.25


class PinholeCamera:
    """Intrinsics and pose of a MuJoCo camera, and the pixel <-> world projections."""

    def __init__(self, model, data, name=SCENE_CAMERA, width=WIDTH, height=HEIGHT):
        cam = model.camera(name)
        self.width, self.height = width, height
        self.f = 0.5 * height / np.tan(np.radians(cam.fovy[0]) / 2)  # focal length in pixels
        self.pos = data.cam_xpos[cam.id].copy()
        self.R = data.cam_xmat[cam.id].reshape(3, 3).copy()  # columns: camera x, y, z axes in world

    def pixel_rays(self, uv):
        """World-frame ray directions through pixel coordinates uv (N x 2, u right, v down)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        x = (uv[:, 0] - self.width / 2) / self.f
        y = -(uv[:, 1] - self.height / 2) / self.f  # image v grows down, camera y points up
        d_cam = np.stack([x, y, -np.ones_like(x)], axis=1)  # a MuJoCo camera looks along its -z
        return d_cam @ self.R.T

    def pixels_to_plane(self, uv, z):
        """Where the rays through pixels uv hit the horizontal plane at height z (N x 3)."""
        d = self.pixel_rays(uv)
        t = (z - self.pos[2]) / d[:, 2]
        return self.pos + t[:, None] * d

    def project(self, points):
        """World points (N x 3) to pixel coordinates (N x 2)."""
        p_cam = (np.atleast_2d(points) - self.pos) @ self.R
        u = self.width / 2 + self.f * p_cam[:, 0] / -p_cam[:, 2]
        v = self.height / 2 - self.f * p_cam[:, 1] / -p_cam[:, 2]
        return np.stack([u, v], axis=1)


def red_mask(rgb):
    """Pixels that look like the red cube: saturated and red in hue (OpenCV hue runs 0..180)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 6) | (h >= 174)) & (s >= 150) & (v >= 60)


def largest_blob(mask):
    """The largest connected region of a boolean mask (or an empty mask)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == best


def top_face(rgb, blob):
    """The brightest part of the cube blob: the top face, lit from above.

    The light shines straight down, so the top face is clearly brighter than the sides.
    Otsu's method picks the brightness threshold between the two groups automatically.
    """
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    vals = value[blob]
    thresh, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return largest_blob(blob & (value > thresh))


def outline_pixels(mask):
    """Pixel coordinates of the mask's outer boundary, pushed out to where the true edge lies."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len).reshape(-1, 2).astype(float) + 0.5  # pixel centres
    centre = pts.mean(axis=0)
    out = pts - centre
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return pts + EDGE_PUSH * out


def estimate_from_image(rgb, camera, iterations=6):
    """Cube pose from one RGB image: dict(pos, yaw, half, pixels) or None if the cube isn't visible."""
    blob = largest_blob(red_mask(rgb))
    if blob.sum() < MIN_PIXELS:
        return None
    face = top_face(rgb, blob)
    if face.sum() < MIN_PIXELS // 2:
        return None
    uv = outline_pixels(face)
    half = CUBE_HALF  # first guess: the nominal cube size
    for _ in range(iterations):
        pts = camera.pixels_to_plane(uv, 2 * half)[:, :2]  # top face lies at z = cube height
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32) * 1000)  # in mm for precision
        half = 0.25 * (w + h) / 1000  # a cube: height = width
    yaw = np.radians(angle)
    return dict(pos=np.array([cx / 1000, cy / 1000, half]), yaw=yaw, half=half, pixels=int(blob.sum()))


def plausible(est):
    return est is not None and HALF_RANGE[0] <= est["half"] <= HALF_RANGE[1]


def agree(a, b):
    """Two estimates of the same, still cube agree (yaw compared modulo the cube's 90 deg symmetry)."""
    d_yaw = (a["yaw"] - b["yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.linalg.norm(a["pos"] - b["pos"]) <= AGREE_POS and abs(d_yaw) <= AGREE_YAW


class CameraPerception:
    """perceive(data) for pick_and_place: render the scene camera and estimate the cube pose.

    Returns (position, yaw) like ground_truth, or None if the cube can't be seen.

    Frames are not trusted blindly. Under parallel load this laptop's OpenGL occasionally returned
    a wrong image (one trial in a few hundred: an estimate 77 cm away from the cube), and real
    cameras deliver corrupted or stale frames too. So the camera looks at least twice and accepts
    an estimate only when it is plausible (a cube-sized cube) and two looks agree. `glitches`
    counts the looks that were thrown away; `last` holds the accepted estimate.
    """

    def __init__(self, model, width=WIDTH, height=HEIGHT):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)
        self.last = None
        self.glitches = 0

    def image(self, data):
        self.renderer.update_scene(data, camera=SCENE_CAMERA)
        return self.renderer.render()

    def look(self, data):
        camera = PinholeCamera(self.model, data, width=self.renderer.width, height=self.renderer.height)
        return estimate_from_image(self.image(data), camera)

    def __call__(self, data):
        self.last = None
        prev, unseen = None, 0
        for _ in range(MAX_LOOKS):
            est = self.look(data)
            if est is None:
                unseen += 1
                if unseen >= 2:  # twice nothing: the cube really is out of view or occluded
                    return None
                continue
            if not plausible(est):
                self.glitches += 1
                continue
            if prev is not None and agree(prev, est):
                self.last = est
                return est["pos"].copy(), est["yaw"]
            if prev is not None:
                self.glitches += 1
            prev = est
        return None

    def close(self):
        self.renderer.close()
