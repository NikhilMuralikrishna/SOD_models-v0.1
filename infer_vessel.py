#!/usr/bin/env python3
"""
Vessel detection inference.

--input takes either a single image or a folder of images.


USAGE
-----

# whole folder
python infer_vessel.py --weights models/vessel_sar_optical/weights.pt \
                       --input test_images/ --out results/vessel \
                       --fork /path/to/folder-containing-yolov10

# one image
python infer_vessel.py --weights models/vessel_sar_optical/weights.pt \
                       --input test_images/scene_01.png --out results/vessel \
                       --fork /path/to/folder-containing-yolov10

# the smaller, faster model
python infer_vessel.py --weights models/vessel_sar_optical_light/weights.pt ...

# georeferenced GeoTIFF input -> also writes detections.geojson
python infer_vessel.py ... --geojson

# verbose: weights path and per-detection confidences
python infer_vessel.py ... -v

# other options
#   --conf 0.25     confidence threshold for keeping a detection
#   --imgsz 640     inference size (640 is what the model was trained at)

--fork is the folder CONTAINING yolov10, not yolov10 itself. These weights do
not load under stock ultralytics.


OUTPUT (written to --out)
-------------------------
    <name>_det.jpg        the image with boxes drawn on it, labelled "ship"
    detections.json       per image: box count and [x1, y1, x2, y2, confidence]
    detections.geojson    with --geojson: EPSG:4326 FeatureCollection, one
                          polygon per detection, with centroid lon/lat


INPUT
-----
PNG, JPG or TIF. 3-channel, 8-bit. One vessel class ("ship").
The same weights serve both SAR and optical - the model was trained on them
mixed, so no modality argument is needed.
"""

import argparse
import json
import os
import sys

def load_model(weights, fork):
    for p in (fork, os.path.join(fork, "yolov10")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        try:
            from ultralytics import YOLOv10 as Y
        except ImportError:
            from ultralytics import YOLO as Y
    except ImportError as e:
        sys.exit(
            f"Could not import the YOLO class: {e}\n"
            f"These weights need the yolov10 fork (Galdelli et al.). Point --fork at the\n"
            f"folder CONTAINING 'yolov10', and make sure huggingface_hub is installed."
        )
    try:
        return Y(weights)
    except ModuleNotFoundError as e:
        sys.exit(f"Loading failed: {e}\nThe fork is not on the path. Check --fork.")


def geo_features(path, boxes, stem):
    """
    Turn pixel boxes into GeoJSON features in EPSG:4326.
    Returns [] if the file carries no CRS.
    """
    try:
        import rasterio
        from rasterio.warp import transform as warp
    except ImportError:
        return []
    with rasterio.open(path) as src:
        if src.crs is None:
            return []
        feats = []
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes, 1):
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            xs, ys = zip(*(src.transform * c for c in corners))
            lons, lats = warp(src.crs, "EPSG:4326", list(xs), list(ys))
            ring = [[lon, lat] for lon, lat in zip(lons, lats)]
            ring.append(ring[0])

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            mx, my = src.transform * (cx, cy)
            clon, clat = warp(src.crs, "EPSG:4326", [mx], [my])

            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "detection_id": f"{stem}-{i:04d}",
                    "source_image": os.path.basename(path),
                    "class": "vessel",
                    "confidence": conf,
                    "centroid_lon": clon[0],
                    "centroid_lat": clat[0],
                    "bbox_pixel": [x1, y1, x2, y2],
                },
            })
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="results/vessel")
    ap.add_argument("--fork", required=True,
                    help="folder CONTAINING the yolov10 folder")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--geojson", action="store_true",
                    help="also write detections.geojson (needs georeferenced input)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print weights path and per-detection confidences")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    if os.path.isdir(a.input):
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        files = sorted(os.path.join(a.input, f) for f in os.listdir(a.input)
                       if f.lower().endswith(exts))
    else:
        files = [a.input]
    if not files:
        sys.exit(f"No images found in {a.input}")

    model = load_model(a.weights, a.fork)
    print(f"\n[vessel] {len(files)} image(s)")
    if a.verbose:
        print(f"  {a.weights}  classes={model.names}")

    all_results = {}
    geo_features_all = []
    ungeoreferenced = []
    total = 0

    for f in files:
        r = model.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0]]
            boxes.append([x1, y1, x2, y2, round(float(b.conf[0]), 4)])

        name = os.path.splitext(os.path.basename(f))[0]
        out_img = os.path.join(a.out, f"{name}_det.jpg")
        try:
            # labels read just "ship" - no confidence number on the image
            arr = r.plot(conf=False)
            from PIL import Image
            Image.fromarray(arr[..., ::-1]).save(out_img)     # plot() returns BGR
        except Exception:                                     # noqa: BLE001
            r.save(filename=out_img)                          # fallback

        all_results[os.path.basename(f)] = {
            "n_detections": len(boxes), "boxes_xyxy_conf": boxes}

        if a.geojson and boxes:
            feats = geo_features(f, boxes, name)
            if feats:
                geo_features_all.extend(feats)
            else:
                ungeoreferenced.append(os.path.basename(f))

        total += len(boxes)
        n = len(boxes)
        line = f"  {os.path.basename(f):40s} " + (
            f"{n} ship" + ("" if n == 1 else "s") if n else "-")
        if a.verbose and boxes:
            line += "   conf " + ", ".join(f"{b[4]:.2f}" for b in boxes[:8])
        print(line)

    with open(os.path.join(a.out, "detections.json"), "w") as fh:
        json.dump(all_results, fh, indent=2)

    if a.geojson:
        if geo_features_all:
            fc = {"type": "FeatureCollection",
                  "crs": {"type": "name",
                          "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                  "features": geo_features_all}
            gp = os.path.join(a.out, "detections.geojson")
            with open(gp, "w") as fh:
                json.dump(fc, fh, indent=2)
            print(f"\n{len(geo_features_all)} georeferenced detections -> {gp}")
        else:
            print("\nNo georeferenced output: no input carried a CRS.")
        if ungeoreferenced:
            print(f"  {len(ungeoreferenced)} image(s) had detections but no CRS, "
                  f"e.g. {ungeoreferenced[0]}")

    print(f"\n{total} ship(s) across {len(files)} image(s) -> {a.out}")


if __name__ == "__main__":
    main()
