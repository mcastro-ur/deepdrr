# -*- coding: utf-8 -*-
import os
import numpy as np
import imageio.v2 as iio
import trimesh
import deepdrr
import deepdrr.geo as geo
from deepdrr.projector import Projector
from deepdrr.vol import Mesh
from deepdrr.pyrenderdrr import DRRMaterial
import cv2
from scipy.ndimage import gaussian_filter

# ---- Headless GL ----
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

# ---- Paths ----
CT_PATH   = "/scratch/mcastro/deepdrr/ct_full.nii.gz"
PLY_PATH  = "/scratch/mcastro/deepdrr/data/6.5mmD_32mmThread_L130mm_1.ply"
FINAL_STL = "/scratch/mcastro/deepdrr/data/mesh_final_strategy_c.stl"
VTK_BBOX_PATH = "/scratch/mcastro/deepdrr/data/outline.vtk"

# ============================================================
# PARAMETRES C-ARM
# ============================================================
SDD       = 1020.0   # Source-to-Detector Distance (mm)
SID       = 530.0    # Source-to-Isocenter Distance (mm)
ALPHA_DEG = 0.0      # LAO(+) / RAO(-) degrés
BETA_DEG  = 0.0      # CRA(+) / CAU(-) degrés
SPECTRUM  = "60KV_AL35"  # Spectre basse énergie = métal très absorbant (noir prononcé)
IUB       = 4.0          # intensity_upper_bound réduit = métal prend plus de place dans l'échelle
# ============================================================

# ---- Taille de sortie ----
OUTPUT_SIZE = (640, 640)  # (largeur, hauteur) en pixels

# ---- Bounding box : couleur et épaisseur ----
BBOX_COLOR     = (0, 255, 0)   # vert vif (BGR)
BBOX_THICKNESS = 1             # épaisseur des arêtes en pixels
DRAW_MARGIN_FACTOR = 2         # tolérance hors image pour ne pas tracer des segments aberrants


def _get_matrix(transform):
    if hasattr(transform, "matrix"):
        return np.array(transform.matrix, dtype=np.float64)
    return np.array(transform, dtype=np.float64)


def _apply_transform(W4x4, vertices):
    ones = np.ones((len(vertices), 1), dtype=np.float64)
    vh = np.hstack([vertices.astype(np.float64), ones])
    return (W4x4 @ vh.T).T[:, :3]


def build_registered_mesh_stl(ct):
    tm = trimesh.load(PLY_PATH, force="mesh")
    W = _get_matrix(ct.world_from_anatomical)
    flip_lps_ras = np.diag([-1., -1., 1., 1.])
    tm.vertices = _apply_transform(W @ flip_lps_ras, tm.vertices)
    tm.export(FINAL_STL)
    mesh_centroid = tm.vertices.mean(axis=0)
    print(f"Mesh centroid (world): {mesh_centroid}")
    return FINAL_STL, mesh_centroid


def load_vtk_bbox_points(vtk_path, ct):
    """
    Charge les points 3D de la bounding box depuis un fichier VTK ASCII
    et les transforme dans le repère world DeepDRR (même stratégie C que le mesh).

    Returns:
        pts_world: np.ndarray (N, 3) en coordonnées world DeepDRR
        lines: list of (i, j) paires d'indices pour les segments à dessiner
    """
    pts = []
    lines = []

    with open(vtk_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    points_match = re.search(r'POINTS\s+(\d+)\s+\w+\s*([\s\S]*?)(?=\n\s*\n|\nMETADATA|\nLINES)', content)
    if points_match:
        n_pts = int(points_match.group(1))
        pts_text = points_match.group(2).strip()
        nums = [float(x) for x in pts_text.split()]
        if len(nums) < n_pts * 3:
            raise ValueError(f"VTK incomplet: {len(nums)} coordonnées pour {n_pts} points")
        for i in range(0, n_pts * 3, 3):
            pts.append([nums[i], nums[i + 1], nums[i + 2]])

    pts = np.array(pts, dtype=np.float64)

    conn_match = re.search(r'CONNECTIVITY\s+\w+\s*([\s\S]*?)(?=\nCELL_DATA|\nPOINT_DATA|\Z)', content)
    if conn_match:
        conn_nums = [int(x) for x in conn_match.group(1).strip().split()]
        for i in range(0, len(conn_nums) - 1, 2):
            lines.append((conn_nums[i], conn_nums[i + 1]))

    W = _get_matrix(ct.world_from_anatomical)
    flip_lps_ras = np.diag([-1., -1., 1., 1.])
    pts_world = _apply_transform(W @ flip_lps_ras, pts)

    print(f"BBox VTK: {len(pts_world)} points, {len(lines)} segments chargés")
    print(f"BBox world bounds: {pts_world.min(axis=0)} → {pts_world.max(axis=0)}")

    return pts_world, lines


def project_bbox_on_image(pts_world, lines, device, img_shape):
    """
    Projette les points 3D world sur l'image 2D via la CameraProjection du C-arm.

    Returns:
        pts_2d: np.ndarray (N, 2) coordonnées pixel (col, row) — peut être hors image
        lines: liste de paires d'indices valides à dessiner
    """
    proj = device.get_camera_projection()

    pts_2d = []
    for pt in pts_world:
        p2d = proj @ geo.point(*pt)
        pts_2d.append([float(p2d[0]), float(p2d[1])])

    pts_2d = np.array(pts_2d, dtype=np.float64)
    return pts_2d, lines


def draw_bbox_on_image(img_u8, pts_2d, lines, color=(0, 255, 0), thickness=2):
    """
    Dessine la bounding box (segments) sur une image uint8 en niveaux de gris.
    L'image d'entrée est supposée déjà à OUTPUT_SIZE (640x640).

    Args:
        img_u8: np.ndarray (H, W) uint8 — déjà redimensionnée à OUTPUT_SIZE
        pts_2d: np.ndarray (N, 2) coordonnées pixel (col, row) dans l'espace natif capteur
        lines: list of (i, j) paires d'indices
        color: couleur BGR des segments
        thickness: épaisseur des segments en pixels

    Returns:
        img_rgb: np.ndarray (OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3) uint8 avec bbox dessinée
    """
    # img_u8 est déjà à OUTPUT_SIZE après imwrite_resized — on travaille directement dessus
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    h_out, w_out = img_u8.shape  # = OUTPUT_SIZE[1], OUTPUT_SIZE[0]

    # Les pts_2d sont dans l'espace capteur natif (ex: 1536x1536).
    # On les met à l'échelle vers OUTPUT_SIZE.
    # On estime la taille native depuis les coordonnées elles-mêmes (max observé).
    # Si les pts sont tous dans [0, OUTPUT_SIZE] c'est déjà OK, sinon on scale.
    if len(pts_2d) > 0:
        native_max_x = pts_2d[:, 0].max()
        native_max_y = pts_2d[:, 1].max()
        native_min_x = pts_2d[:, 0].min()
        native_min_y = pts_2d[:, 1].min()

        # Heuristique : si les coordonnées dépassent largement OUTPUT_SIZE,
        # on suppose qu'elles sont dans l'espace capteur natif et on scale.
        # Sinon on les utilise telles quelles.
        native_w = max(native_max_x, w_out)
        native_h = max(native_max_y, h_out)
        scale_x = w_out / native_w if native_w > w_out else 1.0
        scale_y = h_out / native_h if native_h > h_out else 1.0
    else:
        scale_x = scale_y = 1.0

    margin = max(h_out, w_out) * DRAW_MARGIN_FACTOR

    for i, j in lines:
        px1 = int(round(pts_2d[i, 0] * scale_x))
        py1 = int(round(pts_2d[i, 1] * scale_y))
        px2 = int(round(pts_2d[j, 0] * scale_x))
        py2 = int(round(pts_2d[j, 1] * scale_y))

        p1 = (px1, py1)
        p2 = (px2, py2)

        if (-margin <= p1[0] <= w_out + margin and -margin <= p1[1] <= h_out + margin and
                -margin <= p2[0] <= w_out + margin and -margin <= p2[1] <= h_out + margin):
            cv2.line(img_rgb, p1, p2, color, thickness)

    return img_rgb


def resize_to_output(img):
    """Redimensionne une image (uint8 2D ou 3D) à OUTPUT_SIZE avec interpolation bicubique."""
    return cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def imwrite_resized(path, img):
    """Redimensionne l'image à OUTPUT_SIZE puis l'enregistre (gère grayscale et BGR→RGB)."""
    out = resize_to_output(img)
    if out.ndim == 3:
        # cv2 travaille en BGR ; imageio attend RGB
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    iio.imwrite(path, out)


# ============================================================
#  UTILITAIRES TONE MAPPING
# ============================================================

def reinhard_tonemap(x, white_point=1.0, black_lift=0.0):
    """Tone mapping Reinhard étendu avec compression des blancs."""
    x = np.clip(x, 0.0, 1.0)
    wp2 = white_point * white_point
    x_tm = x * (1.0 + x / wp2) / (1.0 + x)
    if black_lift > 0.0:
        x_tm = x_tm * (1.0 - black_lift) + black_lift
    return np.clip(x_tm, 0.0, 1.0)


# ============================================================
#  PIPELINE FLUOROSCOPIE REALISTE
# ============================================================

def fluoro_realistic(
    img,
    p_low          = 0.5,
    p_high         = 99.5,
    black_lift     = 0.02,
    photons        = 8000.0,
    gamma          = 0.50,
    scatter_sigma  = 30.0,
    scatter_weight = 0.08,
    blur_sigma     = 0.6,
    elec_sigma     = 0.003,
    vignette       = 0.10,
    clahe_clip     = 1.5,
    clahe_grid     = (16, 16),
    invert         = True,
    post_black_lift     = 0.15,
    post_white_compress = 0.92,
    seed           = 42,
):
    rng = np.random.default_rng(seed)
    x = img.astype(np.float64)

    p_lo_val, p_hi_val = np.percentile(x, [p_low, p_high])
    dyn_range = p_hi_val - p_lo_val
    if dyn_range < 1e-6:
        x = np.zeros_like(x)
    else:
        x = np.clip((x - p_lo_val) / dyn_range, 0.0, 1.0)

    if black_lift > 0.0:
        x = x * (1.0 - black_lift) + black_lift

    x = np.power(np.clip(x, 1e-8, 1.0), gamma)

    scatter = gaussian_filter(x.astype(np.float32), sigma=scatter_sigma)
    x = (1.0 - scatter_weight) * x + scatter_weight * scatter.astype(np.float64)

    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), blur_sigma).astype(np.float64)

    lam = np.clip(x * photons, 0, None)
    x = rng.poisson(lam).astype(np.float64) / photons

    x += rng.normal(0.0, elec_sigma, x.shape)
    x = np.clip(x, 0.0, 1.0)

    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xx - w / 2.0) / (w / 2.0)
    yn = (yy - h / 2.0) / (h / 2.0)
    r2 = xn * xn + yn * yn
    vig = 1.0 - vignette * r2
    x *= np.clip(vig, 1.0 - vignette, 1.0)
    x = np.clip(x, 0.0, 1.0)

    if invert:
        x = 1.0 - x

    x = x * (1.0 - post_black_lift) + post_black_lift
    x = reinhard_tonemap(x, white_point=post_white_compress, black_lift=0.0)
    x = np.clip(x, 0.0, 1.0)

    u8 = (x * 255).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    u8 = clahe_obj.apply(u8)

    return u8


def fluoro_high_dose(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5, p_high=99.5, black_lift=0.02,
        photons=12000.0, gamma=0.48,
        scatter_sigma=35.0, scatter_weight=0.07,
        blur_sigma=0.5, elec_sigma=0.002, vignette=0.08,
        clahe_clip=1.3, clahe_grid=(16, 16),
        post_black_lift=0.28, post_white_compress=0.85,
        seed=seed,
    )


def fluoro_standard(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5, p_high=99.5, black_lift=0.02,
        photons=5000.0, gamma=0.52,
        scatter_sigma=28.0, scatter_weight=0.09,
        blur_sigma=0.7, elec_sigma=0.004, vignette=0.12,
        clahe_clip=1.6, clahe_grid=(16, 16),
        post_black_lift=0.25, post_white_compress=0.88,
        seed=seed,
    )


def fluoro_low_dose(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5, p_high=99.5, black_lift=0.02,
        photons=1500.0, gamma=0.55,
        scatter_sigma=22.0, scatter_weight=0.11,
        blur_sigma=0.9, elec_sigma=0.008, vignette=0.15,
        clahe_clip=1.8, clahe_grid=(12, 12),
        post_black_lift=0.20, post_white_compress=0.90,
        seed=seed,
    )


def to_uint8(x):
    x = x.astype(np.float32)
    return (255 * (x - x.min()) / (x.ptp() + 1e-8)).astype(np.uint8)


def to_uint8_shared(a, b):
    vmin = min(float(a.min()), float(b.min()))
    vmax = max(float(a.max()), float(b.max()))
    def _cvt(x):
        return (255 * np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)).astype(np.uint8)
    return _cvt(a), _cvt(b)


def render_pair(ct, mesh, device, tag, bbox_pts_world=None, bbox_lines=None):
    with Projector([ct], device=device,
                   spectrum=SPECTRUM,
                   intensity_upper_bound=IUB,
                   mode="linear", step=0.1) as p0:
        img0 = p0()
    with Projector([ct, mesh], device=device,
                   spectrum=SPECTRUM,
                   intensity_upper_bound=IUB,
                   mode="linear", step=0.1) as p1:
        img1 = p1()

    diff = np.abs(img1.astype(np.float32) - img0.astype(np.float32))
    print(f"[{tag}] diff max={float(diff.max()):.6f} mean={float(diff.mean()):.6f}")

    # ---- DRR bruts ----
    ct_u8, mix_u8 = to_uint8_shared(img0, img1)
    imwrite_resized(f"drr_ct_{tag}.png", ct_u8)
    imwrite_resized(f"drr_mesh_{tag}.png", mix_u8)
    imwrite_resized(f"drr_diff_{tag}.png", to_uint8(diff))

    # ---- Haute dose ----
    f_hd_ct   = fluoro_high_dose(img0)
    f_hd_mesh = fluoro_high_dose(img1)
    imwrite_resized(f"fluoro_hd_ct_{tag}.png", f_hd_ct)
    imwrite_resized(f"fluoro_hd_mesh_{tag}.png", f_hd_mesh)
    imwrite_resized(f"fluoro_hd_diff_{tag}.png",
                    to_uint8(np.abs(f_hd_mesh.astype(np.float32) - f_hd_ct.astype(np.float32))))

    # ---- Dose standard ----
    f_st_ct   = fluoro_standard(img0)
    f_st_mesh = fluoro_standard(img1)
    imwrite_resized(f"fluoro_std_ct_{tag}.png", f_st_ct)
    imwrite_resized(f"fluoro_std_mesh_{tag}.png", f_st_mesh)
    imwrite_resized(f"fluoro_std_diff_{tag}.png",
                    to_uint8(np.abs(f_st_mesh.astype(np.float32) - f_st_ct.astype(np.float32))))

    # ---- Faible dose ----
    f_ld_ct   = fluoro_low_dose(img0)
    f_ld_mesh = fluoro_low_dose(img1)
    imwrite_resized(f"fluoro_ld_ct_{tag}.png", f_ld_ct)
    imwrite_resized(f"fluoro_ld_mesh_{tag}.png", f_ld_mesh)
    imwrite_resized(f"fluoro_ld_diff_{tag}.png",
                    to_uint8(np.abs(f_ld_mesh.astype(np.float32) - f_ld_ct.astype(np.float32))))

    # ---- BBox overlay ----
    if bbox_pts_world is not None:
        pts_2d, bbox_lines_valid = project_bbox_on_image(
            bbox_pts_world, bbox_lines, device, img0.shape
        )

        # Redimensionner d'abord à OUTPUT_SIZE, puis dessiner la bbox à l'échelle
        st_mesh_640  = resize_to_output(f_st_mesh)
        st_ct_640    = resize_to_output(f_st_ct)
        hd_mesh_640  = resize_to_output(f_hd_mesh)
        mix_u8_640   = resize_to_output(mix_u8)

        # draw_bbox_on_image reçoit l'image déjà à 640x640 et scale les pts_2d
        bbox_std_mesh = draw_bbox_on_image(st_mesh_640,  pts_2d, bbox_lines_valid,
                                           color=BBOX_COLOR, thickness=BBOX_THICKNESS)
        bbox_std_ct   = draw_bbox_on_image(st_ct_640,    pts_2d, bbox_lines_valid,
                                           color=BBOX_COLOR, thickness=BBOX_THICKNESS)
        bbox_hd_mesh  = draw_bbox_on_image(hd_mesh_640,  pts_2d, bbox_lines_valid,
                                           color=BBOX_COLOR, thickness=BBOX_THICKNESS)
        bbox_drr_mesh = draw_bbox_on_image(mix_u8_640,   pts_2d, bbox_lines_valid,
                                           color=BBOX_COLOR, thickness=BBOX_THICKNESS)

        # Sauvegarder en RGB (imageio attend RGB, cv2 produit BGR)
        iio.imwrite(f"fluoro_std_mesh_bbox_{tag}.png",
                    cv2.cvtColor(bbox_std_mesh, cv2.COLOR_BGR2RGB))
        iio.imwrite(f"fluoro_std_ct_bbox_{tag}.png",
                    cv2.cvtColor(bbox_std_ct,   cv2.COLOR_BGR2RGB))
        iio.imwrite(f"fluoro_hd_mesh_bbox_{tag}.png",
                    cv2.cvtColor(bbox_hd_mesh,  cv2.COLOR_BGR2RGB))
        iio.imwrite(f"drr_mesh_bbox_{tag}.png",
                    cv2.cvtColor(bbox_drr_mesh, cv2.COLOR_BGR2RGB))
        print(f"[{tag}] BBox overlay saved ({OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} px)")

    print(f"[{tag}] Saved: drr + fluoro_hd + fluoro_std + fluoro_ld  "
          f"({OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} px)")


def make_device(mesh_centroid_world, alpha_deg, beta_deg, sdd, sid):
    """
    Crée un MobileCArm avec :
      - isocentre = centroïde du mesh (position réelle dans world)
      - alpha = LAO(+) / RAO(-) en degrés
      - beta  = CRA(+) / CAU(-) en degrés
      - sdd   = source-to-detector distance (mm)
      - sid   = source-to-isocenter distance (mm)
    """
    device = deepdrr.MobileCArm(
        source_to_detector_distance=sdd,
        source_to_isocenter_vertical_distance=sid,
        alpha=alpha_deg,
        beta=beta_deg,
        degrees=True,
        min_alpha=-180,
        max_alpha=180,
        min_beta=-225,
        max_beta=225,
        enforce_isocenter_bounds=False,
    )
    device.move_to(
        isocenter_in_world=geo.point(*mesh_centroid_world),
        degrees=True,
    )
    src_w = np.array(device.world_from_device @ device.source_in_device)
    iso_w = np.array(device.isocenter_in_world)
    print(f"Source (world)      : {src_w}")
    print(f"Isocentre (world)   : {iso_w}")
    print(f"Dist source-iso     : {np.linalg.norm(src_w - iso_w):.1f} mm  (SID={sid})")
    print(f"Alpha={alpha_deg}°  Beta={beta_deg}°  SDD={sdd} mm")
    return device


def main():
    ct = deepdrr.Volume.from_nifti(CT_PATH)
    print("CT center (world):", np.array(ct.center_in_world))

    stl_path, mesh_centroid = build_registered_mesh_stl(ct)

    mesh = Mesh.from_stl(
        stl_path,
        material=DRRMaterial("titanium", density=14.0),
    )

    device = make_device(
        mesh_centroid_world=mesh_centroid,
        alpha_deg=ALPHA_DEG,
        beta_deg=BETA_DEG,
        sdd=SDD,
        sid=SID,
    )

    bbox_pts_world, bbox_lines = None, None
    if os.path.exists(VTK_BBOX_PATH):
        try:
            bbox_pts_world, bbox_lines = load_vtk_bbox_points(VTK_BBOX_PATH, ct)
        except Exception as e:
            print(f"[WARN] BBox VTK non chargé: {e}")

    render_pair(
        ct,
        mesh,
        device,
        "final",
        bbox_pts_world=bbox_pts_world,
        bbox_lines=bbox_lines,
    )

    print("\nDone. Outputs par preset :")
    print("  drr_ct / drr_mesh / drr_diff")
    print("  fluoro_hd_ct / fluoro_hd_mesh / fluoro_hd_diff   ← haute dose, plus net")
    print("  fluoro_std_ct / fluoro_std_mesh / fluoro_std_diff ← dose standard")
    print("  fluoro_ld_ct / fluoro_ld_mesh / fluoro_ld_diff   ← faible dose, plus granulaire")
    print(f"  Toutes les images : {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} pixels")
    print("  + overlay bbox VTK: drr_mesh_bbox / fluoro_std_ct_bbox / fluoro_std_mesh_bbox / fluoro_hd_mesh_bbox")


if __name__ == "__main__":
    main()
