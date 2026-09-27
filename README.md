# SLID Explorer

An in-browser viewer for a leakage-aware baseline trained on the public **SLID** slit-lamp image dataset.
Upload an anterior-segment photograph and see:

- a **screening score** for any abnormality, with its decision threshold;
- **13 lesion scores**, flagged when above a threshold set at 90% specificity;
- **evidence maps** (exact class activation maps) for each score, with an opacity control;
- an **out-of-distribution warning** when the image does not resemble SLID photographs;
- a **model card** with grouped cross-validation metrics and known limitations.

**Privacy:** inference runs locally with ONNX Runtime Web (WebAssembly). Images never leave the device.

> **Research use only.** Not a medical device. Not clinically validated. Do not use for diagnosis or treatment.

## Run locally

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Deploy on GitHub Pages

Settings → Pages → Source: *Deploy from a branch* → `main` / `(root)`. The site is static; no build step.

## Model

| Item | Value |
|---|---|
| Backbone | EfficientNet-B0, ImageNet weights (timm `ra-3dd342df`), frozen |
| Heads | 14 L2 logistic regressions (C = 0.05, class-balanced), standardisation folded into the weights |
| Input | 384 × 288 RGB; whole frame for SLID-shaped images (4:3 or 1.24:1), centre-crop to 4:3 otherwise |
| Outputs | `probs` (1×14), `cams` (1×14×9×12), `embedding` (1×1280) |
| Validation | 5-fold StratifiedGroupKFold on 1,187 eye groups rebuilt from image similarity (SLID has no patient IDs) |
| Screening AUROC | 0.978 (out-of-fold) |
| Macro lesion AUROC | 0.927 (out-of-fold) |

Per-lesion AUROC, sensitivity at 90% specificity and thresholds are in `model/model_meta.json` and on the page.
Scores are **not calibrated probabilities**; compare each with its own threshold.

The browser pipeline reproduces the training preprocessing (long side to 512, then area resampling to 384 × 288).
On the example images, browser scores match the Python pipeline within 0.02.

The ONNX model and the WebAssembly runtime are stored in parts (`*.partN`) so each file stays small; `app.js` joins them at load time. To rebuild the full model: `cat model/slid_effb0.onnx.part* > slid_effb0.onnx`.

`tools/export.py` rebuilds the ONNX model and metadata from cached embeddings; `tools/pick.py` selects the gallery images.

## Limitations

- One source dataset; behaviour on other cameras and populations is unknown.
- SLID contains acquisition shortcuts (image resolution, annotation shape, file order) that the model may exploit.
- Pinguecula and corneal scarring are detected poorly (sensitivity < 55% at 90% specificity).
- The 12 gallery examples were part of the training data.

## Citation

B. Ben Ammar. *Shortcuts and leakage in public slit-lamp benchmarks.* Manuscript submitted, 2026.

M. Xu, Y. Sun, H. Cheng, Y. Zhou, N. Maimaiti, P. Chen, et al. SLID: a slit-lamp image dataset for deep learning-based anterior eye anatomical segmentation and multi-lesion detection. *Front. Digit. Health* 7 (2026) 1716501.

## Licences

Code: MIT. Example images: SLID, CC BY 4.0 (downscaled to 512 px). ONNX Runtime Web: MIT.
