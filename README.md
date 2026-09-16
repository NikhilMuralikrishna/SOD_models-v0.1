# Vessel Detection and Oil Spill Segmentation Models



This repository contains trained models and inference scripts for two maritime Earth observation tasks:

* **Vessel detection** from SAR and optical Satellite images using YOLOv10.
* **Oil spill segmentation** from SAR or optical Satellite images using a multi-encoder U-Net.



## Repository structure

``` text
ai4copsec\\\_univpm-models-v0.1/
├── README.md
├── config.py
├── model.py
├── infer\\\_oilspill.py
├── infer\\\_vessel.py
├── requirements.txt
└── models/
    ├── vessel\\\_sar\\\_optical/
    │   └── weights.pt
    ├── vessel\\\_sar\\\_optical\\\_light/
    │   └── weights.pt
    ├── oilspill\\\_sar/
    │   └── weights.pth
    └── oilspill\\\_optical/
        └── weights.pth
```

## Requirements

Python 3.9 or a compatible environment is recommended.

The oil spill pipeline requires PyTorch, NumPy, Pillow and Rasterio.
Rasterio is required for GeoTIFF input and geographic outputs.

The vessel detector requires the compatible YOLOv10 fork. The supplied
vessel weights are not intended for use with stock Ultralytics alone.

Install the supplied dependencies with:

``` bash
pip install -r requirements.txt
```

Install the YOLOv10 fork according to its own requirements before
running vessel inference.

\---

# Oil spill segmentation

This contains trained weights for **SAR** and **optical** inference.

No thermal checkpoint is provided.

The inference script determines the modality from the
number of input bands:

Modality     Bands Required order

\---

SAR              2 VH, VV
Optical          4 B2, B3, B4, B8

SAR inputs are expected to contain Sigma0 values in dB. Optical inputs
must contain the four bands in the order shown above.

The script accepts GeoTIFF (`.tif`, `.tiff`) and NumPy (`.npy`) files in
`(C,H,W)` or `(H,W,C)` layout. Arrays are converted to `float32`.

Per-channel z-score normalisation is applied automatically:

``` text
(channel - channel\\\_mean) / (channel\\\_std + 1e-8)
```

Use `--no-normalise` only if the input is already normalised in the same
way.

Images with dimensions divisible by 16 can be processed directly. Other
dimensions are zero-padded to the next multiple of 16 for inference, and
the prediction is cropped back to the original image size.

The current inference configuration records reference sizes of 256 × 256
for SAR and 240 × 240 for optical. Other compatible sizes can still be
processed.

## Running oil spill inference

Single file:

``` bash
python infer\\\_oilspill.py --input "path/to/image.tif" --out "results/oilspill"
```

Folder containing compatible SAR and optical files:

``` bash
python infer\\\_oilspill.py --input "path/to/tiles" --out "results/oilspill"
```

## Oil spill outputs

For every processed input:

``` text
<name>\\\_mask.png
<name>\\\_overlay.png
summary.json
```

`summary.json` records the modality, image size, predicted oil-pixel
count and fraction, and probability statistics.

### Georeferenced inputs

If an input GeoTIFF contains a valid CRS and geotransform, the script
also writes:

``` text
<name>\\\_mask.tif
<name>\\\_mask.geojson
```

The mask GeoTIFF retains the source CRS and transform and can be opened
directly in GIS software such as QGIS.

The GeoJSON contains vector polygons generated from connected predicted
oil regions. Each region includes a unique identifier, centroid
longitude and latitude, and an approximate area in square metres.

```

\\---

# Vessel detection

Vessel detection uses YOLOv10. Two checkpoints are supplied:

\\---

Model                               Weights

\\---

Vessel detector                     `models/vessel\\\_sar\\\_optical/weights.pt`

Vessel detector Lightweight         `models/vessel\\\_sar\\\_optical\\\_light/weights.pt`



Both checkpoints detect one class: \*\*ship\*\*.

The same weights are used for the supported SAR and optical vessel
imagery.

Input images are expected to be 3-channel, 8-bit PNG, JPG or TIF images.
The default inference size is 640.

## YOLOv10 dependency

`--fork` must point to the directory \*\*containing\*\* the `yolov10`
directory.

For example:

``` text
release/
├── infer\\\_vessel.py
└── yolov10/
```

When running from `release/`, use:

``` text
--fork "."
```

## Running vessel inference

Full model:

``` bash
python infer\\\_vessel.py --weights "models/vessel\\\_sar\\\_optical/weights.pt" --input "path/to/images" --out "results/vessel" --fork "."
```

Lightweight model:

``` bash
python infer\\\_vessel.py --weights "models/vessel\\\_sar\\\_optical\\\_light/weights.pt" --input "path/to/images" --out "results/vessel\\\_light" --fork "."
```

## Vessel outputs

The script writes:

``` text
<name>\\\_det.jpg
detections.json
```

The annotated image shows detected ships with bounding boxes.

`detections.json` stores each detection as:

``` text
\\\[x1, y1, x2, y2, confidence]
```

where `(x1,y1)` and `(x2,y2)` are the upper-left and lower-right corners
of the bounding box in image pixel coordinates.

For example:

``` json
{
  "scene.jpg": {
    "n\\\_detections": 1,
    "boxes\\\_xyxy\\\_conf": \\\[
      \\\[167.0, 131.1, 231.2, 149.5, 0.335]
    ]
  }
}
```

This represents one retained vessel detection with a confidence of
0.335.

### Georeferenced vessel detections

For a georeferenced GeoTIFF, add:

``` text
--geojson
```

The script then creates `detections.geojson`. Each detection is
represented as a geographic polygon derived from its image-space
bounding box. Properties include the source image, class, confidence,
centroid longitude and latitude, and original pixel bounding box.

If the input does not contain a CRS, normal vessel detection still runs,
but geographic coordinates cannot be generated.

\---

.

## 

