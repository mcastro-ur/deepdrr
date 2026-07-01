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

# ============================================================
# PARAMETRES C-ARM
# ============================================================
SDD       = 1020.0   # Source-to-Detector Distance (mm)
SID       = 530.0    # Source-to-Isocenter Distance (mm)
ALPHA_DEG = 0.0      # LAO(+) / RAO(-) degrés
BETA_DEG  = 0.0      # CRA(+) / CAU(-) degrés
# ============================================================

# ---- Taille de sortie ----
OUTPUT_SIZE = (640, 640)  # (largeur, hauteur) en pixels


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


def resize_to_output(img):
    """Redimensionne une image (uint8 2D) à OUTPUT_SIZE avec interpolation bicubique."""
    return cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def imwrite_resized(path, img):
    """Redimensionne l'image à OUTPUT_SIZE puis l'enregistre."""
    iio.imwrite(path, resize_to_output(img))


# ============================================================
#  UTILITAIRES TONE MAPPING
# ============================================================

def reinhard_tonemap(x, white_point=1.0, black_lift=0.0):
    """Tone mapping Reinhard étendu avec compression des blancs.

    Args:
        x: image float [0, 1]
        white_point: valeur au-dessus de laquelle on compresse
        black_lift: relève les noirs (0 = pas de relevé)

    Returns:
        image tone-mappée, float [0, 1]
    """
    x = np.clip(x, 0.0, 1.0)
    # Reinhard étendu : x * (1 + x / white_point^2) / (1 + x)
    wp2 = white_point * white_point
    x_tm = x * (1.0 + x / wp2) / (1.0 + x)
    # Relevé des noirs
    if black_lift > 0.0:
        x_tm = x_tm * (1.0 - black_lift) + black_lift
    return np.clip(x_tm, 0.0, 1.0)


# ============================================================
#  PIPELINE FLUOROSCOPIE REALISTE
#  Basé sur la chaîne physique d'un vrai flat-panel C-arm :
#  DRR (atténuation) → normalisation douce → relevé noirs →
#  gamma doux → scatter basse-freq → flou détecteur →
#  bruit quantique (Poisson haute dose) → bruit électronique →
#  vignettage subtil → inversion → relevé noirs post-inv →
#  compression blancs → CLAHE très doux
# ============================================================

def fluoro_realistic(
    img,
    # --- Normalisation percentile ---
    p_low          = 0.5,    # percentile bas  (0.5 = préserve la dynamique complète)
    p_high         = 99.5,   # percentile haut (99.5 = préserve la dynamique complète)
    # --- Relevé des noirs PRE-inversion (rôle mineur) ---
    black_lift     = 0.02,   # relève légèrement les noirs avant inversion
    # --- Exposition ---
    photons        = 8000.0,  # haute dose => quasi pas de grain
    # --- Gamma (chaîne DICOM typique) ---
    gamma          = 0.50,    # 0.45-0.55 = courbe typique flat-panel
    # --- Scatter (diffusion Compton basse fréquence) ---
    scatter_sigma  = 30.0,    # large = basse fréquence réaliste
    scatter_weight = 0.08,    # 8% scatter/primaire : typique thorax/bassin
    # --- Flou détecteur (MTF flat-panel ~0.3-0.6 mm FWHM) ---
    blur_sigma     = 0.6,     # en pixels, très léger
    # --- Bruit électronique (readout noise) ---
    elec_sigma     = 0.003,   # très faible sur flat-panel moderne
    # --- Vignettage ---
    vignette       = 0.10,    # subtil, 10% perte aux coins
    # --- CLAHE très doux ---
    clahe_clip     = 1.5,     # faible = pas d'artefacts de contraste local
    clahe_grid     = (16, 16), # grande grille = transition douce
    # --- Inversion ---
    invert         = True,
    # --- Relevé des noirs POST-inversion ---
    post_black_lift     = 0.15,   # relève les noirs APRES inversion (0.10-0.20)
    post_white_compress = 0.92,   # compresse légèrement les blancs post-inversion
    seed           = 42,
):
    rng = np.random.default_rng(seed)
    x = img.astype(np.float64)

    # 1. Normalisation percentile robuste (préserve la dynamique)
    p_lo_val, p_hi_val = np.percentile(x, [p_low, p_high])
    dyn_range = p_hi_val - p_lo_val
    if dyn_range < 1e-6:
        # Image quasi-uniforme : pas de normalisation utile
        x = np.zeros_like(x)
    else:
        x = np.clip((x - p_lo_val) / dyn_range, 0.0, 1.0)

    # 2. Relevé des noirs PRE-inversion (rôle mineur, évite noirs purs)
    if black_lift > 0.0:
        x = x * (1.0 - black_lift) + black_lift

    # 3. Gamma (courbe de réponse détecteur)
    x = np.power(np.clip(x, 1e-8, 1.0), gamma)

    # 4. Scatter basse fréquence (Compton diffus)
    #    Utiliser scipy pour éviter artefacts bord OpenCV
    scatter = gaussian_filter(x.astype(np.float32), sigma=scatter_sigma)
    x = (1.0 - scatter_weight) * x + scatter_weight * scatter.astype(np.float64)

    # 5. Flou détecteur (MTF flat-panel)
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), blur_sigma).astype(np.float64)

    # 6. Bruit quantique Poisson (dose haute => grain très faible)
    lam = np.clip(x * photons, 0, None)
    x = rng.poisson(lam).astype(np.float64) / photons

    # 7. Bruit électronique (readout, très faible sur flat-panel)
    x += rng.normal(0.0, elec_sigma, x.shape)

    x = np.clip(x, 0.0, 1.0)

    # 8. Vignettage subtil (chute illumination aux bords)
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xx - w / 2.0) / (w / 2.0)
    yn = (yy - h / 2.0) / (h / 2.0)
    r2 = xn * xn + yn * yn
    vig = 1.0 - vignette * r2
    x *= np.clip(vig, 1.0 - vignette, 1.0)

    x = np.clip(x, 0.0, 1.0)

    # 9. Inversion (os = blanc, fond = noir → style fluoro)
    if invert:
        x = 1.0 - x

    # ---- Post-inversion : relevé des noirs + compression des blancs ----
    # Relève les noirs (zones sombres après inversion = fond/tissu mou)
    x = x * (1.0 - post_black_lift) + post_black_lift
    # Compression douce des blancs (évite saturation os/métal)
    x = reinhard_tonemap(x, white_point=post_white_compress, black_lift=0.0)
    x = np.clip(x, 0.0, 1.0)

    # 10. CLAHE très doux (rehaussement local minimal)
    u8 = (x * 255).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    u8 = clahe_obj.apply(u8)

    return u8


# --- Preset haute dose (chirurgie, bonne visibilité métal) ---
def fluoro_high_dose(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5,
        p_high=99.5,
        black_lift=0.02,
        photons=12000.0,
        gamma=0.48,
        scatter_sigma=35.0,
        scatter_weight=0.07,
        blur_sigma=0.5,
        elec_sigma=0.002,
        vignette=0.08,
        clahe_clip=1.3,
        clahe_grid=(16, 16),
        post_black_lift=0.18,
        post_white_compress=0.90,
        seed=seed,
    )


# --- Preset dose standard (intervention orthopédique typique) ---
def fluoro_standard(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5,
        p_high=99.5,
        black_lift=0.02,
        photons=5000.0,
        gamma=0.52,
        scatter_sigma=28.0,
        scatter_weight=0.09,
        blur_sigma=0.7,
        elec_sigma=0.004,
        vignette=0.12,
        clahe_clip=1.6,
        clahe_grid=(16, 16),
        post_black_lift=0.15,
        post_white_compress=0.92,
        seed=seed,
    )


# --- Preset faible dose (pédiatrique / réduction exposition) ---
def fluoro_low_dose(img, seed=42):
    return fluoro_realistic(
        img,
        p_low=0.5,
        p_high=99.5,
        black_lift=0.02,
        photons=1500.0,
        gamma=0.55,
        scatter_sigma=22.0,
        scatter_weight=0.11,
        blur_sigma=0.9,
        elec_sigma=0.008,
        vignette=0.15,
        clahe_clip=1.8,
        clahe_grid=(12, 12),
        post_black_lift=0.12,
        post_white_compress=0.93,
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


def render_pair(ct, mesh, device, tag):
    with Projector([ct], device=device, intensity_upper_bound=12.0, mode="linear", step=0.1) as p0:
        img0 = p0()
    with Projector([ct, mesh], device=device, intensity_upper_bound=12.0, mode="linear", step=0.1) as p1:
        img1 = p1()

    diff = np.abs(img1.astype(np.float32) - img0.astype(np.float32))
    print(f"[{tag}] diff max={float(diff.max()):.6f} mean={float(diff.mean()):.6f}")

    # DRR bruts (normalisation partagée)
    ct_u8, mix_u8 = to_uint8_shared(img0, img1)
    imwrite_resized(f"drr_ct_{tag}.png",   ct_u8)
    imwrite_resized(f"drr_mesh_{tag}.png", mix_u8)
    imwrite_resized(f"drr_diff_{tag}.png", to_uint8(diff))

    # --- Haute dose ---
    f_hd_ct   = fluoro_high_dose(img0)
    f_hd_mesh = fluoro_high_dose(img1)
    imwrite_resized(f"fluoro_hd_ct_{tag}.png",   f_hd_ct)
    imwrite_resized(f"fluoro_hd_mesh_{tag}.png",  f_hd_mesh)
    imwrite_resized(f"fluoro_hd_diff_{tag}.png",
                to_uint8(np.abs(f_hd_mesh.astype(np.float32) - f_hd_ct.astype(np.float32))))

    # --- Dose standard ---
    f_st_ct   = fluoro_standard(img0)
    f_st_mesh = fluoro_standard(img1)
    imwrite_resized(f"fluoro_std_ct_{tag}.png",   f_st_ct)
    imwrite_resized(f"fluoro_std_mesh_{tag}.png",  f_st_mesh)
    imwrite_resized(f"fluoro_std_diff_{tag}.png",
                to_uint8(np.abs(f_st_mesh.astype(np.float32) - f_st_ct.astype(np.float32))))

    # --- Faible dose ---
    f_ld_ct   = fluoro_low_dose(img0)
    f_ld_mesh = fluoro_low_dose(img1)
    imwrite_resized(f"fluoro_ld_ct_{tag}.png",   f_ld_ct)
    imwrite_resized(f"fluoro_ld_mesh_{tag}.png",  f_ld_mesh)
    imwrite_resized(f"fluoro_ld_diff_{tag}.png",
                to_uint8(np.abs(f_ld_mesh.astype(np.float32) - f_ld_ct.astype(np.float32))))

    print(f"[{tag}] Saved: drr + fluoro_hd + fluoro_std + fluoro_ld  ({OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} px)")


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
        # density=7.7 g/cm³ : valeur effective pour implant titane/acier chirurgical
        # (titane pur ~4.5, alliage Ti-6Al-4V ~4.43, acier inox ~8.0)
        # ajuster selon le matériau réel de l'implant
        material=DRRMaterial("titanium", density=7.7),
    )

    device = make_device(
        mesh_centroid_world=mesh_centroid,
        alpha_deg=ALPHA_DEG,
        beta_deg=BETA_DEG,
        sdd=SDD,
        sid=SID,
    )

    render_pair(ct, mesh, device, "final")

    print("\nDone. Outputs par preset :")
    print("  drr_ct / drr_mesh / drr_diff")
    print("  fluoro_hd_ct / fluoro_hd_mesh / fluoro_hd_diff   ← haute dose, plus net")
    print("  fluoro_std_ct / fluoro_std_mesh / fluoro_std_diff ← dose standard")
    print("  fluoro_ld_ct / fluoro_ld_mesh / fluoro_ld_diff   ← faible dose, plus granulaire")
    print(f"  Toutes les images : {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} pixels")


if __name__ == "__main__":
    main()
