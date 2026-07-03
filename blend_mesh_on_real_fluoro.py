# -*- coding: utf-8 -*-
"""
Pipeline : mesh positionné via CT (stratégie C) + blend sur vraies images fluoro JPEG.

Le CT est utilisé UNIQUEMENT pour récupérer world_from_anatomical et positionner
le mesh dans le bon repère world DeepDRR. Ensuite le blend se fait sur les vraies images.

Usage :
    python blend_mesh_on_real_fluoro.py
"""
import os
import re
import glob
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

# ============================================================
# CHEMINS
# ============================================================
CT_PATH         = "/scratch/mcastro/deepdrr/ct_full.nii.gz"
PLY_PATH        = "/scratch/mcastro/deepdrr/data/6.5mmD_32mmThread_L130mm_1.ply"
FINAL_STL       = "/scratch/mcastro/deepdrr/data/mesh_blend_only.stl"
REAL_FLUORO_DIR = "/scratch/mcastro/deepdrr/real_fluoro/"   # dossier JPEGs réels
OUTPUT_DIR      = "/scratch/mcastro/deepdrr/blended/"       # sortie
VTK_BBOX_PATH   = "/scratch/mcastro/deepdrr/data/outline.vtk"

# ============================================================
# PARAMETRES C-ARM
# ============================================================
SDD         = 1020.0   # Source-to-Detector Distance (mm)
SID         = 530.0    # Source-to-Isocenter Distance (mm)
ALPHA_DEG   = 0.0      # LAO(+) / RAO(-) degrés
BETA_DEG    = 0.0      # CRA(+) / CAU(-) degrés
PIXEL_SIZE  = 0.194    # mm/pixel (Siemens CIOS Fusion)
SENSOR_SIZE = 1536     # pixels (capteur carré)

# ============================================================
# PARAMETRES BLEND
# ============================================================
OUTPUT_SIZE    = (640, 640)
BLEND_STRENGTH = 0.92         # 0=invisible, 1=métal pur noir
METAL_DENSITY  = 14.0         # g/cm³
SPECTRUM       = "60KV_AL35"  # basse énergie = métal très absorbant
IUB_METAL      = 3.0          # intensity_upper_bound mesh seul
EDGE_SIGMA     = 1.2          # lissage bords du masque métal (pénombre X-ray)


# ============================================================
# UTILITAIRES
# ============================================================
def _get_matrix(transform):
    if hasattr(transform, "matrix"):
        return np.array(transform.matrix, dtype=np.float64)
    return np.array(transform, dtype=np.float64)


def _apply_transform(W4x4, vertices):
    ones = np.ones((len(vertices), 1), dtype=np.float64)
    vh = np.hstack([vertices.astype(np.float64), ones])
    return (W4x4 @ vh.T).T[:, :3]


# ============================================================
# ETAPE 1 : Positionner le mesh via le CT (stratégie C)
# Identique à example_projector_claude_v3.py
# ============================================================
def build_registered_mesh_stl(ct):
    """
    Charge le PLY et applique la transformation world_from_anatomical + flip LPS→RAS.
    Identique à la stratégie C de example_projector_claude_v3.py.
    Le CT est utilisé UNIQUEMENT pour récupérer world_from_anatomical.
    """
    tm = trimesh.load(PLY_PATH, force="mesh")
    W = _get_matrix(ct.world_from_anatomical)
    flip_lps_ras = np.diag([-1., -1., 1., 1.])
    tm.vertices = _apply_transform(W @ flip_lps_ras, tm.vertices)
    tm.export(FINAL_STL)
    mesh_centroid = tm.vertices.mean(axis=0)
    print(f"Mesh centroid (world): {mesh_centroid}")
    return FINAL_STL, mesh_centroid


# ============================================================
# ETAPE 2 : Créer le C-arm virtuel
# ============================================================
def make_device(mesh_centroid_world, alpha_deg=ALPHA_DEG, beta_deg=BETA_DEG):
    device = deepdrr.MobileCArm(
        source_to_detector_distance=SDD,
        source_to_isocenter_vertical_distance=SID,
        alpha=alpha_deg,
        beta=beta_deg,
        degrees=True,
        pixel_size=PIXEL_SIZE,
        sensor_height=SENSOR_SIZE,
        sensor_width=SENSOR_SIZE,
        min_alpha=-180, max_alpha=180,
        min_beta=-225,  max_beta=225,
        enforce_isocenter_bounds=False,
    )
    device.move_to(
        isocenter_in_world=geo.point(*mesh_centroid_world),
        degrees=True,
    )
    src_w = np.array(device.world_from_device @ device.source_in_device)
    iso_w = np.array(device.isocenter_in_world)
    print(f"Source (world)    : {src_w}")
    print(f"Isocentre (world) : {iso_w}")
    print(f"Dist source-iso   : {np.linalg.norm(src_w - iso_w):.1f} mm")
    print(f"Alpha={alpha_deg}°  Beta={beta_deg}°  SDD={SDD} mm")
    return device


# ============================================================
# ETAPE 3 : DRR mesh SEUL (sans CT dans le Projector)
# ============================================================
def render_mesh_only(mesh, device):
    """
    Projette uniquement le mesh — pas de CT.
    
    Returns:
        metal_att: float32 array (H, W) [0,1] — carte atténuation métal
                   0 = air, 1 = métal dense
    """
    with Projector(
        [mesh],               # <-- mesh seul, pas de CT
        device=device,
        spectrum=SPECTRUM,
        intensity_upper_bound=IUB_METAL,
        mode="linear",
        step=0.1,
    ) as p:
        drr_raw = p()         # neglog déjà appliqué → [0, 1]

    metal_att = drr_raw.astype(np.float32)
    # Normalise par le max pour avoir [0, 1] propre
    max_val = metal_att.max()
    if max_val > 1e-6:
        metal_att = metal_att / max_val
    metal_att = np.clip(metal_att, 0.0, 1.0)

    nonzero = drr_raw[drr_raw > 0.01]
    print(f"  DRR métal — max={drr_raw.max():.4f} "
          f"pixels_metal={len(nonzero)} ({100*len(nonzero)/drr_raw.size:.1f}%)")
    return metal_att


# ============================================================
# ETAPE 4 : Blend métal sur image réelle
# ============================================================
def blend_metal_on_fluoro(real_bgr, metal_att,
                           blend_strength=BLEND_STRENGTH,
                           output_size=OUTPUT_SIZE,
                           edge_sigma=EDGE_SIGMA):
    """
    Fusionne la carte d'atténuation métal sur l'image fluoro réelle.
    
    Physique X-ray : le métal absorbe → zones métal plus sombres.
    result = real * (1 - metal_att * blend_strength)
    
    Args:
        real_bgr    : image BGR uint8 (taille quelconque)
        metal_att   : carte atténuation [0,1] taille native capteur
        blend_strength : intensité du blend
        output_size : taille finale (W, H)
        edge_sigma  : lissage bords masque (pénombre X-ray réaliste)
    
    Returns:
        result_bgr : BGR uint8 à output_size
    """
    h_real, w_real = real_bgr.shape[:2]

    # Redimensionner carte métal à la taille de l'image réelle
    metal_r = cv2.resize(metal_att, (w_real, h_real), interpolation=cv2.INTER_LINEAR)

    # Lisser les bords (pénombre X-ray, évite bords durs artificiels)
    if edge_sigma > 0:
        metal_r = gaussian_filter(metal_r.astype(np.float32), sigma=edge_sigma)
    metal_r = np.clip(metal_r, 0.0, 1.0)

    # Blend physique
    real_f = real_bgr.astype(np.float32) / 255.0
    metal_3ch = metal_r[:, :, np.newaxis]
    result_f = real_f * (1.0 - metal_3ch * blend_strength)
    result_f = np.clip(result_f, 0.0, 1.0)

    result_bgr = (result_f * 255).astype(np.uint8)

    # Redimensionner à output_size
    result_bgr = cv2.resize(result_bgr, output_size, interpolation=cv2.INTER_CUBIC)
    return result_bgr


# ============================================================
# BBOX VTK OVERLAY (optionnel)
# ============================================================
def load_vtk_bbox_points(vtk_path, ct):
    """Identique à example_projector_claude_v3.py"""
    pts = []
    lines = []
    with open(vtk_path, "r", encoding="utf-8") as f:
        content = f.read()

    points_match = re.search(
        r'POINTS\s+(\d+)\s+\w+\s*([\s\S]*?)(?=\n\s*\n|\nMETADATA|\nLINES)', content
    )
    if points_match:
        n_pts = int(points_match.group(1))
        nums = [float(x) for x in points_match.group(2).strip().split()]
        for i in range(0, n_pts * 3, 3):
            pts.append([nums[i], nums[i+1], nums[i+2]])
    pts = np.array(pts, dtype=np.float64)

    conn_match = re.search(
        r'CONNECTIVITY\s+\w+\s*([\s\S]*?)(?=\nCELL_DATA|\nPOINT_DATA|\Z)', content
    )
    if conn_match:
        conn_nums = [int(x) for x in conn_match.group(1).strip().split()]
        for i in range(0, len(conn_nums) - 1, 2):
            lines.append((conn_nums[i], conn_nums[i+1]))

    W = _get_matrix(ct.world_from_anatomical)
    flip_lps_ras = np.diag([-1., -1., 1., 1.])
    pts_world = _apply_transform(W @ flip_lps_ras, pts)

    print(f"BBox VTK: {len(pts_world)} points, {len(lines)} segments")
    return pts_world, lines


def draw_bbox_on_bgr(img_bgr_640, pts_world, lines, device,
                     color=(0, 255, 0), thickness=2):
    """Projette et dessine la bbox sur image BGR 640×640."""
    proj = device.get_camera_projection()
    h, w = img_bgr_640.shape[:2]

    pts_2d = []
    for pt in pts_world:
        p2d = proj @ geo.point(*pt)
        pts_2d.append([float(p2d[0]), float(p2d[1])])
    pts_2d = np.array(pts_2d)

    # Scale depuis espace capteur natif vers output_size
    if len(pts_2d) > 0:
        native_w = max(pts_2d[:, 0].max(), w)
        native_h = max(pts_2d[:, 1].max(), h)
        sx = w / native_w
        sy = h / native_h
    else:
        sx = sy = 1.0

    result = img_bgr_640.copy()
    margin = max(h, w) * 3
    for i, j in lines:
        p1 = (int(round(pts_2d[i, 0] * sx)), int(round(pts_2d[i, 1] * sy)))
        p2 = (int(round(pts_2d[j, 0] * sx)), int(round(pts_2d[j, 1] * sy)))
        if (abs(p1[0]) < margin and abs(p1[1]) < margin and
                abs(p2[0]) < margin and abs(p2[1]) < margin):
            cv2.line(result, p1, p2, color, thickness)
    return result


# ============================================================
# TRAITEMENT D'UNE IMAGE
# ============================================================
def process_one_image(real_jpeg_path, mesh, device, output_dir,
                       bbox_pts_world=None, bbox_lines=None):
    basename = os.path.splitext(os.path.basename(real_jpeg_path))[0]
    print(f"\n--- Traitement : {basename} ---")

    # Charger image réelle
    real_bgr = cv2.imread(real_jpeg_path)
    if real_bgr is None:
        print(f"  [WARN] Impossible de lire {real_jpeg_path}")
        return

    # DRR mesh seul (carte atténuation métal)
    metal_att = render_mesh_only(mesh, device)

    # Blend sur image réelle
    blended_bgr = blend_metal_on_fluoro(real_bgr, metal_att)

    # Image réelle redimensionnée (référence)
    real_640 = cv2.resize(real_bgr, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    # DRR métal seul (debug)
    drr_u8 = (metal_att * 255).astype(np.uint8)
    drr_640 = cv2.resize(drr_u8, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    os.makedirs(output_dir, exist_ok=True)

    # Sauvegardes
    # 1. Image réelle seule
    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_real.jpg"),
        real_640,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # 2. DRR métal seul (debug)
    iio.imwrite(
        os.path.join(output_dir, f"{basename}_drr_metal.png"),
        drr_640
    )

    # 3. Résultat blend
    iio.imwrite(
        os.path.join(output_dir, f"{basename}_blended.png"),
        cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    )

    # 4. Blend + bbox overlay
    if bbox_pts_world is not None:
        blended_bbox = draw_bbox_on_bgr(
            blended_bgr, bbox_pts_world, bbox_lines, device
        )
        iio.imwrite(
            os.path.join(output_dir, f"{basename}_blended_bbox.png"),
            cv2.cvtColor(blended_bbox, cv2.COLOR_BGR2RGB)
        )

    print(f"  → {basename}_real.jpg / _drr_metal.png / _blended.png / _blended_bbox.png")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Etape 1 : CT uniquement pour la transformation spatiale ---
    print("Chargement CT (pour transformation spatiale uniquement)...")
    ct = deepdrr.Volume.from_nifti(CT_PATH)
    print(f"CT center (world): {np.array(ct.center_in_world)}")

    # Positionner le mesh avec la stratégie C (identique à example_projector_claude_v3.py)
    stl_path, mesh_centroid = build_registered_mesh_stl(ct)

    # Le CT n'est plus utilisé après cette étape
    del ct
    print("CT libéré — mesh positionné correctement.")

    # --- Etape 2 : Créer mesh et C-arm ---
    mesh = Mesh.from_stl(
        stl_path,
        material=DRRMaterial("titanium", density=METAL_DENSITY),
    )

    device = make_device(
        mesh_centroid_world=mesh_centroid,
        alpha_deg=ALPHA_DEG,
        beta_deg=BETA_DEG,
    )

    # --- Etape 3 : BBox VTK (optionnel) ---
    bbox_pts_world, bbox_lines = None, None
    if os.path.exists(VTK_BBOX_PATH):
        try:
            # Recharger CT juste pour la bbox
            ct_for_bbox = deepdrr.Volume.from_nifti(CT_PATH)
            bbox_pts_world, bbox_lines = load_vtk_bbox_points(VTK_BBOX_PATH, ct_for_bbox)
            del ct_for_bbox
        except Exception as e:
            print(f"[WARN] BBox non chargée: {e}")

    # --- Etape 4 : Traiter toutes les images JPEG ---
    jpeg_files = sorted(
        glob.glob(os.path.join(REAL_FLUORO_DIR, "*.jpg")) +
        glob.glob(os.path.join(REAL_FLUORO_DIR, "*.jpeg")) +
        glob.glob(os.path.join(REAL_FLUORO_DIR, "*.png"))
    )

    if not jpeg_files:
        print(f"[WARN] Aucune image trouvée dans {REAL_FLUORO_DIR}")
        print(f"       Placer les JPEGs dans : {REAL_FLUORO_DIR}")
        return

    print(f"\nTrouvé {len(jpeg_files)} image(s) à traiter")

    for jpeg_path in jpeg_files:
        process_one_image(
            jpeg_path, mesh, device, OUTPUT_DIR,
            bbox_pts_world=bbox_pts_world,
            bbox_lines=bbox_lines,
        )

    print(f"\n✓ Done. Résultats dans : {OUTPUT_DIR}")
    print(f"  Fichiers par image :")
    print(f"    *_real.jpg          → image originale redimensionnée 640×640")
    print(f"    *_drr_metal.png     → DRR mesh seul (debug)")
    print(f"    *_blended.png       → résultat final (vraie fluoro + mesh)")
    print(f"    *_blended_bbox.png  → résultat + bounding box orientée")


if __name__ == "__main__":
    main()
