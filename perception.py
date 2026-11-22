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

from scene import CUBE_HALF, MARKER_SIZE, MARKER_Z, MARKERS, SCENE_CAMERA

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
    """Intrinsics and pose of a camera, and the pixel <-> world projections.

    Pixel convention: u to the right, v down, and the centre of pixel (i, j) is at (i + 0.5, j + 0.5).
    PinholeCamera(model, data) reads the true camera from the simulator; from_pose builds one from a
    believed pose (a stale calibration, or the pose estimated from the ArUco marker).
    """

    def __init__(self, model, data, name=SCENE_CAMERA, width=WIDTH, height=HEIGHT):
        cam = model.camera(name)
        self.width, self.height = width, height
        self.f = 0.5 * height / np.tan(np.radians(cam.fovy[0]) / 2)  # focal length in pixels
        self.pos = data.cam_xpos[cam.id].copy()
        self.R = data.cam_xmat[cam.id].reshape(3, 3).copy()  # columns: camera x, y, z axes in world

    @classmethod
    def from_pose(cls, pos, R, f, width=WIDTH, height=HEIGHT):
        cam = cls.__new__(cls)
        cam.pos, cam.R, cam.f, cam.width, cam.height = np.asarray(pos, float), np.asarray(R, float), f, width, height
        return cam

    def K(self):
        """OpenCV camera matrix (in OpenCV's convention, pixel centres are at integer coordinates)."""
        return np.array([[self.f, 0, self.width / 2 - 0.5], [0, self.f, self.height / 2 - 0.5], [0, 0, 1]])

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


# --- the camera locating itself from the ArUco marker ------------------------------------------

_DETECTOR = None


def marker_detector():
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # corners to a fraction of a pixel
        _DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    return _DETECTOR


def marker_corners_world(marker_id):
    """A tag's black-square corners in world coordinates, in ArUco's order: top-left, top-right,
    bottom-right, bottom-left of the printed image. The texture lies with the image's right = world
    +x and the image's up = world +y (checked against renders: 2 px fit, other orientations ~30 px)."""
    h = MARKER_SIZE / 2
    x, y = MARKERS[marker_id]
    return np.array([[x - h, y + h, MARKER_Z], [x + h, y + h, MARKER_Z], [x + h, y - h, MARKER_Z],
                     [x - h, y - h, MARKER_Z]])


def locate_camera(rgb, f, width=WIDTH, height=HEIGHT, min_markers=1):
    """Camera pose from the ArUco tags in the image, given the (believed) focal length f.

    Returns (PinholeCamera, number of tags used), or (None, n) if fewer than min_markers are visible.
    solvePnP finds the camera rotation and translation that map all visible tags' known corners
    onto their detected pixels. Tags far apart pin the camera's angle down much better than the
    four corners of one small tag.
    """
    corners, ids, _ = marker_detector().detectMarkers(rgb)
    found = [] if ids is None else [(int(i), c) for i, c in zip(ids.ravel(), corners) if int(i) in MARKERS]
    if len(found) < min_markers:
        return None, len(found)
    world = np.vstack([marker_corners_world(i) for i, _ in found])
    pixels = np.vstack([c.reshape(4, 2) for _, c in found]).astype(np.float64)
    K = PinholeCamera.from_pose(np.zeros(3), np.eye(3), f, width, height).K()
    ok, rvec, tvec = cv2.solvePnP(world, pixels, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, len(found)
    rvec, tvec = cv2.solvePnPRefineLM(world, pixels, K, None, rvec, tvec)
    R_cv, _ = cv2.Rodrigues(rvec)  # world -> OpenCV camera frame (x right, y down, z forward)
    pos = -R_cv.T @ tvec.ravel()
    R = R_cv.T @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes: x right, y up, looking along -z
    return PinholeCamera.from_pose(pos, R, f, width, height), len(found)


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
    aspect = min(w, h) / max(w, h)  # 1 for a fully visible top face; a partly hidden one is not square
    return dict(pos=np.array([cx / 1000, cy / 1000, half]), yaw=yaw, half=half, aspect=aspect,
                pixels=int(blob.sum()))


# --- tracking: fit a cube of known size to the whole red silhouette ---------------------------

def cube_corners(x, y, yaw, half):
    """The 8 corners of a cube resting on the table (4 bottom, then 4 top), centre (x, y), turned yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    xy = square @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.c_[xy, np.zeros(4)], np.c_[xy, np.full(4, 2 * half)]])


SUB = 8  # sub-pixel precision for drawing outlines (fixed point with 3 fractional bits)


# Measured while tracking a jittering cube (296 frames): fits with under 30% of the cube visible were
# 43 mm off (median); 30-40%: 4.8 mm but 17 mm at the 90th percentile; 40-50%: 2.2 mm; above: 1-6 mm.
# (A shadowed top face is partly darker than the red threshold, so even a fully seen cube is not 100%.)
MIN_VISIBLE = 0.4


def fit_silhouette(rgb, camera, guess_pos, guess_yaw, half, min_coverage=0.85, min_visible=MIN_VISIBLE,
                   reseed=True):
    """Cube pose from the whole red silhouette, given its size: robust to shadows on the top face.

    While tracking, the gripper's shadow darkens the cube's top face, so "brightest part = top face"
    (estimate_from_image) fails. The silhouette (top and sides together) is still red, and for a
    cube of known size resting on the table it is fixed by (x, y, yaw). Search those three, starting
    from the previous estimate, for the projected outline that best matches the red pixels. Red
    pixels outside the outline cost a lot; outline pixels that are not red cost little, since a
    finger may cover part of the cube. Returns an estimate dict, or None if the fit explains less
    than min_coverage of the red pixels, or less than min_visible of the fitted cube is red (most of
    it hidden: the remaining red pixels are then all "explained", but no longer pin the pose down).
    Yaw is the least precise part (a few degrees): a small turn moves only a few outline pixels.
    """
    mask = largest_blob(red_mask(rgb))
    if mask.sum() < MIN_PIXELS:
        return None
    vs, us = np.nonzero(mask)
    pad = 20
    u0, v0 = max(us.min() - pad, 0), max(vs.min() - pad, 0)
    crop = mask[v0:vs.max() + pad, u0:us.max() + pad]
    n_red = crop.sum()
    canvas = np.zeros(crop.shape, np.uint8)

    def score(p):
        canvas[:] = 0
        px = camera.project(cube_corners(p[0], p[1], p[2], half)) - [u0 + 0.5, v0 + 0.5]
        hull = cv2.convexHull(np.round(px * SUB).astype(np.int32))
        cv2.fillConvexPoly(canvas, hull, 1, lineType=cv2.LINE_8, shift=3)
        model = canvas.astype(bool)
        inside, area = np.count_nonzero(model & crop), np.count_nonzero(model)
        return inside - 3.0 * (n_red - inside) - 0.3 * (area - inside), (inside, area)

    def search(start):
        """Pattern search from `start` = (x, y, yaw): try steps along each axis, halve when stuck."""
        best = np.array(start, float)
        best_score, best_counts = score(best)
        steps = np.array([0.006, 0.006, np.radians(6.0)])  # x, y in m; yaw in rad
        while steps[0] > 0.00025:
            improved = False
            for k in range(3):
                for sign in (1.0, -1.0):
                    trial = best.copy()
                    trial[k] += sign * steps[k]
                    sc, counts = score(trial)
                    if sc > best_score:
                        best, best_score, best_counts, improved = trial, sc, counts, True
            if not improved:
                steps /= 2
        return best, best_score, best_counts

    # Two starts, keep the better fit. The search only looks nearby, so from the previous estimate
    # alone one bad frame could start a cascade: the guess goes stale, the cube moves on, and every
    # later frame starts further away (42-87 mm off in failed trials, 23-56 frames lost in a row,
    # with the cube 43-83% visible). The red blob's centre, projected to the cube's mid-height, is
    # never more than a few mm off and does not depend on earlier frames.
    starts = [(guess_pos[0], guess_pos[1], guess_yaw)]
    if reseed:
        centre = camera.pixels_to_plane(np.array([[us.mean() + 0.5, vs.mean() + 0.5]]), half)[0]
        starts.append((centre[0], centre[1], guess_yaw))
    best, best_score, best_counts = max((search(s) for s in starts), key=lambda r: r[1])
    inside, area = best_counts
    coverage, visible = inside / max(n_red, 1), inside / max(area, 1)
    if coverage < min_coverage or visible < min_visible:
        return None
    return dict(pos=np.array([best[0], best[1], half]), yaw=best[2], half=half, aspect=1.0,
                coverage=coverage, visible=visible, pixels=int(n_red))


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

    `calibration` is what perception believes about the camera:
      ideal   the true camera, read from the simulator (perfect calibration)
      stale   `believed`: a fixed PinholeCamera, e.g. the pose from before the tripod was bumped
      marker  the camera pose located from the ArUco tags in each image, with focal length f_belief
    `belief` holds the camera belief behind the accepted estimate, `tags` how many tags were used.
    """

    def __init__(self, model, width=WIDTH, height=HEIGHT, calibration="ideal", believed=None, f_belief=None):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)
        self.calibration, self.believed, self.f_belief = calibration, believed, f_belief
        self.last = self.belief = None
        self.glitches, self.tags = 0, 0

    def image(self, data):
        self.renderer.update_scene(data, camera=SCENE_CAMERA)
        return self.renderer.render()

    def camera_belief(self, rgb, data):
        w, h = self.renderer.width, self.renderer.height
        if self.calibration == "ideal":
            return PinholeCamera(self.model, data, width=w, height=h)
        if self.calibration == "stale":
            return self.believed
        camera, self.tags = locate_camera(rgb, self.f_belief, w, h)
        return camera

    def look(self, data):
        rgb = self.image(data)
        camera = self.camera_belief(rgb, data)
        est = None if camera is None else estimate_from_image(rgb, camera)
        if est is not None:
            est["camera"] = camera
        return est

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
                self.last, self.belief = est, est["camera"]
                return est["pos"].copy(), est["yaw"]
            if prev is not None:
                self.glitches += 1
            prev = est
        return None

    def close(self):
        self.renderer.close()
