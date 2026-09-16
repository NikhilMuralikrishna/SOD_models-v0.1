# Vessel Detection and Oil Spill Segmentation Models

This repository contains trained models and inference scripts for two maritime Earth observation tasks:

- **Vessel detection** from SAR and optical satellite images using YOLOv10.
- **Oil spill segmentation** from SAR or optical satellite images using a multi-encoder U-Net.

## Repository Structure

```text
SOD_models-v0.1/
├── README.md
├── config.py
├── model.py
├── infer_oilspill.py
├── infer_vessel.py
├── requirements.txt
├── models/
│   ├── vessel_sar_optical/
│   │   └── weights.pt
│   ├── vessel_sar_optical_light/
│   │   └── weights.pt
│   ├── oilspill_sar/
│   │   └── weights.pth
│   └── oilspill_optical/
│       └── weights.pth
└── yolov10/
```

## Requirements

Python 3.9 or a compatible environment is recommended.

The oil spill pipeline requires PyTorch, NumPy, Pillow, and Rasterio. Rasterio is required for GeoTIFF input and geographic outputs.

The vessel detector requires the compatible YOLOv10 fork included in this repository. The supplied vessel weights are not intended for use with stock Ultralytics alone.

Install the supplied dependencies with:

```bash
pip install -r requirements.txt
```

Install the YOLOv10 dependencies according to the requirements provided in the `yolov10/` directory before running vessel inference.

---

# Oil Spill Segmentation

The repository contains trained oil spill segmentation weights for **SAR** and **optical** imagery.

No thermal checkpoint is provided.

The inference script determines the modality from the number of input bands:

| Modality | Bands | Required order |
|---|---:|---|
| SAR | 2 | VH, VV |
| Optical | 4 | B2, B3, B4, B8 |

SAR inputs are expected to contain Sigma0 values in dB. Optical inputs must contain the four bands in the order shown above.

The script accepts GeoTIFF (`.tif`, `.tiff`) and NumPy (`.npy`) files in `(C, H, W)` or `(H, W, C)` layout. Arrays are converted to `float32`.

Per-channel z-score normalisation is applied automatically:

```text
(channel - channel_mean) / (channel_std + 1e-8)
```

Use `--no-normalise` only if the input is already normalised in the same way.

Images with dimensions divisible by 16 can be processed directly. Other dimensions are zero-padded to the next multiple of 16 for inference, and the prediction is cropped back to the original image size.

## Running Oil Spill Inference

Single file:

```bash
python infer_oilspill.py --input "path/to/image.tif" --out "results/oilspill"
```

Folder containing compatible input files:

```bash
python infer_oilspill.py --input "path/to/tiles" --out "results/oilspill"
```

## Oil Spill Outputs

For every processed input:

```text
<name>_mask.png
<name>_overlay.png
summary.json
```

`summary.json` records the modality, image size, predicted oil-pixel count and fraction, and probability statistics.

### Georeferenced Inputs

If an input GeoTIFF contains a valid CRS and geotransform, the script also writes:

```text
<name>_mask.tif
<name>_mask.geojson
```

The mask GeoTIFF retains the source CRS and transform and can be opened directly in GIS software such as QGIS.

The GeoJSON contains vector polygons generated from connected predicted oil regions. Each region includes a unique identifier, centroid longitude and latitude, and an approximate area in square metres.

---

# Vessel Detection

Vessel detection uses YOLOv10. Two checkpoints are supplied:

| Model | Weights |
|---|---|
| Vessel detector | `models/vessel_sar_optical/weights.pt` |
| Lightweight vessel detector | `models/vessel_sar_optical_light/weights.pt` |

Both checkpoints detect one class: **ship**.

The same weights are used for the supported SAR and optical vessel imagery.

Input images are expected to be 3-channel, 8-bit PNG, JPG, or TIF images. The default inference size is 640.

## YOLOv10 Dependency

The compatible `yolov10/` directory is included in this repository.

The `--fork` argument must point to the directory **containing** the `yolov10` directory.

For example:

```text
SOD_models-v0.1/
├── infer_vessel.py
└── yolov10/
```

When running from `SOD_models-v0.1/`, use:

```text
--fork "."
```

## Running Vessel Inference

Full model:

```bash
python infer_vessel.py --weights "models/vessel_sar_optical/weights.pt" --input "path/to/images" --out "results/vessel" --fork "."
```

Lightweight model:

```bash
python infer_vessel.py --weights "models/vessel_sar_optical_light/weights.pt" --input "path/to/images" --out "results/vessel_light" --fork "."
```

## Vessel Outputs

The script writes:

```text
<name>_det.jpg
detections.json
```

The annotated image shows detected ships with bounding boxes.

`detections.json` stores each detection as:

```text
[x1, y1, x2, y2, confidence]
```

where `(x1, y1)` and `(x2, y2)` are the upper-left and lower-right corners of the bounding box in image pixel coordinates.

For example:

```json
{
  "scene.jpg": {
    "n_detections": 1,
    "boxes_xyxy_conf": [
      [167.0, 131.1, 231.2, 149.5, 0.335]
    ]
  }
}
```

This represents one retained vessel detection with a confidence of `0.335`.

### Georeferenced Vessel Detections

For a georeferenced GeoTIFF, add:

```text
--geojson
```

The script then creates:

```text
detections.geojson
```

Each detection is represented as a geographic polygon derived from its image-space bounding box.

If the input does not contain a CRS, normal vessel detection still runs, but geographic coordinates cannot be generated.

