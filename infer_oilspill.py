#!/usr/bin/env python3
"""
Oil spill segmentation inference.

--input takes either a single file or a folder. Modality is detected from the
band count (2 = SAR, 4 = optical) and the matching weights are loaded, so you
never need to say which sensor a file came from.


USAGE
-----

# whole folder, mixed SAR and optical - each file routed automatically
python infer_oilspill.py --input "Test files/Oil Spill" --out results/oilspill

# one file
python infer_oilspill.py --input "Test files/Oil Spill/sar_test.tif" --out results/oilspill

# weights live elsewhere
python infer_oilspill.py --input tiles/ --out results --models /path/to/models

# one specific checkpoint instead of the models/ folder
python infer_oilspill.py --input tile.tif --out results \
                         --weights models/oilspill_sar/weights.pth

# force a modality (files with the wrong band count are skipped)
python infer_oilspill.py --input tiles/ --out results --modality sar

# verbose: checkpoint details and probability statistics
python infer_oilspill.py --input tiles/ --out results -v

# other options
#   --threshold 0.5     probability cut-off for the binary mask
#   --min-area 500      drop polygons under 500 m2 (georeferenced input only)
#   --no-normalise      skip z-scoring; only if tiles are already normalised
#   --model-def PATH    use a different model.py
#   --class-name NAME   class to build (default MultiModalUNet)


OUTPUT (per input file, written to --out)
-----------------------------------------
    <name>_mask.png       binary mask, white = oil
    <name>_overlay.png    channel 1 in grey with the mask in red
    summary.json          modality, size, oil pixel count and fraction, probabilities

If the input GeoTIFF is georeferenced, two more are written:

    <name>_mask.tif       the mask with the input's CRS and transform - opens
                          in QGIS on top of the source scene
    <name>_mask.geojson   one polygon per slick in EPSG:4326, with centroid
                          lon/lat and approximate area in m2

and summary.json gains crs, n_polygons, total_area_m2 and the largest polygon.
Areas are approximate: for a geographic CRS the pixel size in metres is derived
from the latitude, so they are good to a few percent, not survey grade.


INPUT
-----
GeoTIFF or .npy, shaped (C,H,W) or (H,W,C).
    SAR     = 2 channels: ch1 = VH, ch2 = VV, Sigma0 in dB
    Optical = 4 channels: B2, B3, B4, B8 (Blue, Green, Red, NIR)

Any size divisible by 16 runs natively (the model is fully convolutional).
Other sizes are zero-padded to the next multiple of 16 and cropped back
afterwards, so the output always matches the input size. A size differing from
the training size is reported, not blocked.

Per-channel z-score normalisation is applied to match training
(dataset.py _normalise).

The model returns probabilities directly - its decoder already applies
sigmoid. Do not apply sigmoid again.
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

# rasterio warns on every un-georeferenced file; we handle that case explicitly
import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass

# band count -> (modality, weights folder, size it was trained at)
BY_CHANNELS = {
    2: ("sar", "oilspill_sar", 256),
    4: ("optical", "oilspill_optical", 240),
}
CHANNELS = {"sar": 2, "optical": 4}
FOLDER = {v[0]: v[1] for v in BY_CHANNELS.values()}
TRAINED_AT = {v[0]: v[2] for v in BY_CHANNELS.values()}


def load_class(path, name):
    if not os.path.isfile(path):
        sys.exit(f"--model-def not found: {path}")
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)          # model.py imports config.py from here
    spec = importlib.util.spec_from_file_location("usermodel", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        sys.exit(f"Could not import {path}: {e}\nconfig.py must sit next to model.py.")
    if not hasattr(mod, name):
        cand = [n for n in dir(mod) if "net" in n.lower()]
        sys.exit(f"'{name}' not in {path}. Candidates: {cand or 'none'}")
    return getattr(mod, name)


def normalise(image):
    """Per-channel z-score, per image. Matches dataset.py _normalise."""
    out = np.zeros_like(image)
    for c in range(image.shape[0]):
        ch = image[c]
        out[c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out


def read_array(path):
    """Return (C,H,W) float32, or None if unreadable."""
    try:
        if path.lower().endswith(".npy"):
            a = np.load(path)
        else:
            import rasterio
            with rasterio.open(path) as src:
                a = src.read()
    except ImportError:
        sys.exit("rasterio needed for GeoTIFF input (or use .npy)")
    except Exception as e:                                   # noqa: BLE001
        print(f"  skip {os.path.basename(path)}: {str(e)[:80]}")
        return None
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    # (H,W,C) -> (C,H,W) when the last axis is the band axis
    if a.shape[-1] in BY_CHANNELS and a.shape[0] not in BY_CHANNELS:
        a = np.transpose(a, (2, 0, 1))
    return a


def pad_to_16(a):
    """Zero-pad H and W up to the next multiple of 16. Returns (padded, (ph, pw))."""
    _, h, w = a.shape
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="constant")
    return a, (ph, pw)


def geo_info(path):
    """Return (crs, transform) if the file is georeferenced, else (None, None)."""
    if path.lower().endswith(".npy"):
        return None, None
    try:
        import rasterio
        with rasterio.open(path) as src:
            if src.crs is None:
                return None, None
            return src.crs, src.transform
    except Exception:                                       # noqa: BLE001
        return None, None


def pixel_area_m2(transform, crs, lat):
    """Approximate ground area of one pixel, in square metres."""
    dx, dy = abs(transform.a), abs(transform.e)
    if crs.is_geographic:
        import math
        return (dx * 111320.0 * math.cos(math.radians(lat))) * (dy * 110540.0)
    return dx * dy                                          # projected CRS: already metres


def write_geo_outputs(mask, crs, transform, out_dir, name):
    """
    Write a georeferenced mask GeoTIFF and a polygonised GeoJSON in EPSG:4326.
    Returns a list of polygon records for the summary.
    """
    import rasterio
    from rasterio import features
    from rasterio.warp import transform as warp

    with rasterio.open(os.path.join(out_dir, f"{name}_mask.tif"), "w",
                       driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                       count=1, dtype="uint8", crs=crs, transform=transform,
                       nodata=255, compress="lzw") as dst:
        dst.write(mask, 1)

    feats, records = [], []
    for i, (geom, val) in enumerate(
            features.shapes(mask, mask=mask.astype(bool), transform=transform), 1):
        if val != 1:
            continue
        ring = geom["coordinates"][0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

        if str(crs) != "EPSG:4326":
            lon, lat = warp(crs, "EPSG:4326", [cx], [cy])
            clon, clat = lon[0], lat[0]
            rings = []
            for p in ring:
                lo, la = warp(crs, "EPSG:4326", [p[0]], [p[1]])
                rings.append([lo[0], la[0]])
        else:
            clon, clat = cx, cy
            rings = [[p[0], p[1]] for p in ring]

        # pixel count inside this polygon, approximated from its own extent
        px_area = pixel_area_m2(transform, crs, clat)
        shoelace = abs(sum(xs[j] * ys[j + 1] - xs[j + 1] * ys[j]
                           for j in range(len(xs) - 1))) / 2.0
        area_m2 = shoelace / (abs(transform.a) * abs(transform.e)) * px_area

        rec = {"spill_id": f"{name}-{i:04d}",
               "centroid_lon": round(clon, 6),
               "centroid_lat": round(clat, 6),
               "area_m2": round(area_m2, 1)}
        records.append(rec)
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [rings]},
                      "properties": rec})

    if feats:
        fc = {"type": "FeatureCollection",
              "crs": {"type": "name",
                      "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
              "features": feats}
        with open(os.path.join(out_dir, f"{name}_mask.geojson"), "w") as fh:
            json.dump(fc, fh, indent=2)
    return records


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models",
                    help="folder holding oilspill_sar/ and oilspill_optical/")
    ap.add_argument("--weights", help="single checkpoint; overrides --models")
    ap.add_argument("--modality", choices=["sar", "optical"],
                    help="override band-count detection")
    ap.add_argument("--model-def", default=None,
                    help="path to model.py (default: ./model.py beside this script)")
    ap.add_argument("--class-name", default="MultiModalUNet")
    ap.add_argument("--input", required=True,
                    help="a single file, or a folder of files")
    ap.add_argument("--out", default="results/oilspill")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-normalise", action="store_true",
                    help="skip z-score; only if tiles are already normalised")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="drop polygons below this area in m2 (georeferenced input only)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print checkpoint details and probability statistics")
    a = ap.parse_args()

    import torch

    if a.model_def is None:
        here = os.path.dirname(os.path.abspath(__file__))
        a.model_def = os.path.join(here, "model.py")
        if not os.path.isfile(a.model_def):
            sys.exit("model.py not found beside this script. Pass --model-def.")

    os.makedirs(a.out, exist_ok=True)
    ModelClass = load_class(a.model_def, a.class_name)

    if os.path.isdir(a.input):
        files = sorted(sum((glob.glob(os.path.join(a.input, e))
                            for e in ("*.npy", "*.tif", "*.tiff")), []))
    else:
        files = [a.input]
    if not files:
        sys.exit(f"No .npy or .tif found in {a.input}")

    # ---- route each file by band count ----------------------------------
    jobs = {}
    for f in files:
        arr = read_array(f)
        if arr is None:
            continue
        c = arr.shape[0]
        if a.modality:
            if c != CHANNELS[a.modality]:
                print(f"  skip {os.path.basename(f)}: {c} channels, "
                      f"--modality {a.modality} expects {CHANNELS[a.modality]}")
                continue
            modality = a.modality
        elif c in BY_CHANNELS:
            modality = BY_CHANNELS[c][0]
        else:
            print(f"  skip {os.path.basename(f)}: {c} channels, "
                  f"expected 2 (SAR) or 4 (optical)")
            continue
        jobs.setdefault(modality, []).append((f, arr))

    if not jobs:
        sys.exit("Nothing to process.")

    summary = {}

    # ---- one model load per modality ------------------------------------
    for modality, items in jobs.items():
        wp = a.weights or os.path.join(a.models, FOLDER[modality], "weights.pth")
        if not os.path.isfile(wp):
            print(f"  skip {modality}: weights not found at {wp}")
            continue

        ck = torch.load(wp, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        model = ModelClass()
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"\n[{modality}] {len(items)} file(s)")
        if a.verbose:
            print(f"  {wp}")
            print(f"  epoch={ck.get('epoch')}  val_iou={ck.get('val_iou')}  "
                  f"tensors={len(sd)}")

        for f, raw in items:
            h, w = raw.shape[1], raw.shape[2]
            note = ""
            if a.verbose and (h, w) != (TRAINED_AT[modality],) * 2:
                note += f"  [trained at {TRAINED_AT[modality]}]"

            arr = raw if a.no_normalise else normalise(raw)
            padded, (ph, pw) = pad_to_16(arr)
            if a.verbose and (ph or pw):
                note += f"  [padded +{ph},+{pw}]"

            x = torch.from_numpy(padded[None])
            with torch.no_grad():
                prob = model(**{modality: x})
            prob = prob[0, 0].cpu().numpy()          # already sigmoid'd
            if ph or pw:
                prob = prob[:h, :w]                  # crop the padding back off

            mask = (prob >= a.threshold).astype(np.uint8)

            name = os.path.splitext(os.path.basename(f))[0]
            save_png(mask * 255, os.path.join(a.out, f"{name}_mask.png"))

            base = raw[0]
            lo, hi = float(base.min()), float(base.max())
            base = ((base - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
            rgb = np.dstack([base, base, base])
            rgb[mask == 1] = [255, 60, 60]
            save_png(rgb, os.path.join(a.out, f"{name}_overlay.png"))

            entry = {
                "modality": modality,
                "size": [h, w],
                "oil_pixels": int(mask.sum()),
                "oil_fraction": round(float(mask.mean()), 5),
                "mean_prob": round(float(prob.mean()), 4),
                "max_prob": round(float(prob.max()), 4),
            }

            crs, transform = geo_info(f)
            geo_note = ""
            if crs is not None and mask.any():
                try:
                    recs = write_geo_outputs(mask, crs, transform, a.out, name)
                    if a.min_area:
                        recs = [r for r in recs if r["area_m2"] >= a.min_area]
                    entry["crs"] = str(crs)
                    entry["n_polygons"] = len(recs)
                    if recs:
                        entry["total_area_m2"] = round(sum(r["area_m2"] for r in recs), 1)
                        big = max(recs, key=lambda r: r["area_m2"])
                        entry["largest"] = big
                        geo_note = (f"  {len(recs)} polygon(s), "
                                    f"largest at {big['centroid_lat']:.4f}, "
                                    f"{big['centroid_lon']:.4f}")
                except Exception as e:                       # noqa: BLE001
                    geo_note = f"  [geo output failed: {str(e)[:60]}]"
            elif crs is None:
                entry["crs"] = None

            summary[os.path.basename(f)] = entry

            found = "oil" if mask.any() else "-"
            line = f"  {os.path.basename(f):32s} {h}x{w}  {found}"
            if a.verbose:
                line += (f"  {mask.mean()*100:5.2f}%  "
                         f"prob mean {prob.mean():.3f} max {prob.max():.3f}")
            print(line + geo_note + note)

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{len(summary)} of {len(files)} tiles -> {a.out}")


if __name__ == "__main__":
    main()
