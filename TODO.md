# ☀️ High-Precision Solar Filament Instance Segmentation: A Methodological Survey & OPT-HQ Net Architecture

> **Abstract & Survey Overview**: Automated identification and fine-grained instance segmentation of solar filaments from full-disk $2048 \times 2048$ H-Alpha ($H\alpha$) solar observations represent a vital capability for modern space weather forecasting. This document provides a comprehensive methodological survey analyzing computer vision paradigms, bounding box failure modes, and introduces the **Oriented-Prompted Topological Network (OPT-HQ Net)**.

---

## 📌 Table of Contents

1. [Domain Analysis, Dataset Constraints & Mathematical Metrics](#1-domain-analysis-dataset-constraints--mathematical-metrics)
2. [Critical Analysis of Bounding Box Cropping & Overlap Failure Modes](#2-critical-analysis-of-bounding-box-cropping--overlap-failure-modes)
3. [Comprehensive Survey of Computer Vision Paradigms](#3-comprehensive-survey-of-computer-vision-paradigms)
4. [Proposed Architecture: OPT-HQ Net](#4-proposed-architecture-the-oriented-prompted-topological-network-opt-hq-net)
5. [Implementation Strategy, Training Protocols & Post-Processing Pipeline](#5-implementation-strategy-training-protocols--post-processing-pipeline)
6. [Strategic Conclusions & Recommendations](#6-strategic-conclusions--recommendations)

---

## 1. Domain Analysis, Dataset Constraints & Mathematical Metrics

### 1.1 Physical Domain Context
Solar filaments consist of dense, cool plasma clouds suspended in the dynamic solar corona by intense magnetic field lines along **Magnetic Neutral Lines (MNLs)** or **Polarity Inversion Lines (PILs)**. When projected against the solar disk, these structures appear as dark, highly elongated, curvilinear absorption ribbons. Precise delineation of solar filaments is essential because their destabilization and eruption serve as primary precursors to major geomagnetic hazards:
* **Coronal Mass Ejections (CMEs)**
* **Solar Flares**
* **Solar Energetic Particle (SEP) Storms**

### 1.2 The MAGFiLO Benchmark Dataset
The benchmark dataset **MAGFiLO** (*Manually Annotated GONG Filaments from H-Alpha Observations*) provides **10,244** expert-annotated filament instances across **1,593** full-disk observations acquired by the Global Oscillation Network Group (GONG). Benchmarking segmentation algorithms on MAGFiLO presents unique computer vision challenges:

* **Fine-Scale Structural Fidelity**: Filaments feature thin, thread-like lateral structures known as *"barbs"* extending along local magnetic field orientations. Capturing barbs is vital for inferring magnetic field chirality (handedness), yet standard downsampling operations frequently erase these low-contrast features.
* **Background Noise & Imaging Artifacts**: Ground-based $H\alpha$ observations suffer from atmospheric turbulence, vignetting, limb darkening, non-uniform solar illumination, and observatory imaging artifacts. Distinguishing genuine filament plasma from background solar features or cloud shadows requires robust context discrimination.
* **Structural Continuity vs. Over-Merging**: A primary failure mode is *structural fragmentation*, where a contiguous ribbon is segmented into disconnected "island" artifacts. Conversely, models often *over-merge* separate, closely adjacent, or physically crossing filaments into a single composite mask.

### 1.3 Mathematical Formulation of Panoptic Quality (PQ)
Submissions on the MAGFiLO benchmark leaderboard are evaluated primarily via the **Panoptic Quality (PQ)** metric, complemented by Mean Dice score and IoU distributions:

$$\text{PQ}(Y, \hat{Y}) = \frac{\sum_{(y, \hat{y}) \in TP} \text{IoU}(y, \hat{y})}{|TP| + 0.5|FP| + 0.5|FN|}$$

Where:
* $Y$: Set of ground-truth filaments.
* $\hat{Y}$: Set of predicted filaments.
* $TP$ (True Positives): Strictly require spatial overlap of $\text{IoU}(y, \hat{y}) > 0.5$ between matched segment pairs.
* $FP$ (False Positives): Unmatched predicted filaments or redundant split fragments.
* $FN$ (False Negatives): Uncaptured ground-truth instances.

> [!WARNING]
> **Severe Penalties on Structural Errors**:
> * **Fragmentation**: If a model fragments a single contiguous filament into two distinct segments, at most one segment can achieve $\text{IoU} > 0.5$. The second fragment is penalized as an unmatched $FP$, while inflating $FN$ for any under-covered region.
> * **Over-Merging**: If two adjacent filaments are merged into a single predicted mask, the spatial overlap for each individual ground-truth instance drops below $0.5$, triggering **two $FN$ penalties and one $FP$ penalty**.

### 1.4 Benchmark Challenge Mapping

| Challenge Dimension | Physical & Imaging Source | Metric Impact | Technical Requirement |
| :--- | :--- | :--- | :--- |
| 🌿 **Barb Delineation** | Magnetic field lines extending off primary spine | Reduces pixel-level Dice & IoU scores | Multi-scale feature retention & edge enhancement |
| 🌫️ **Background Noise** | Atmospheric turbulence, limb darkening, shadows | Inflates False Positive ($FP$) counts | Context-aware feature filtering & background masking |
| 🧩 **Structural Fragmentation** | Variable optical depth along filament axis | Triggers severe $FP$ and $FN$ penalties in $PQ$ | Topological connectivity constraints & skeleton loss |
| 🔀 **Instance Overlap** | Spatial proximity of parallel or crossing ribbons | Induces many-to-one over-merging penalties | Oriented localized proposals & instance query isolation |

---

## 2. Critical Analysis of Bounding Box Cropping & Overlap Failure Modes

### 2.1 Limitations of Axis-Aligned Horizontal Bounding Boxes (HBB)
A standard two-stage instance segmentation strategy relies on axis-aligned horizontal bounding boxes (HBB) specified by coordinates $(x_c, y_c, w, h)$ to isolate Regions of Interest (RoI). However, solar filaments exhibit extreme aspect ratios (frequently exceeding $10:1$) and orient arbitrarily across the solar disk.

When an elongated filament of length $L$ and width $W$ ($L \gg W$) is oriented diagonally at angle $\theta \approx 45^\circ$, the enclosed HBB area expands dramatically:

$$A_{\text{HBB}} = (L \cdot |\cos\theta| + W \cdot |\sin\theta|) \times (L \cdot |\sin\theta| + W \cdot |\cos\theta|)$$

For $L \gg W$, the area scales as:

$$A_{\text{HBB}} \approx L^2 |\sin\theta \cos\theta|$$

This introduces **up to 90% irrelevant background pixels** into the cropped RoI feature map.

```
       Axis-Aligned (HBB) Clutter                   Oriented Bounding Box (OBB)
   +-------------------------------+             +-----------------------+
   |  Background Clutter (~90%)   /|             | \                     \
   |                             / |             |  \  Filament Ribbon    \
   |   Filament A              /   |             |   \                     \
   |   /\                    /     |   vs.       +----\--------------------+\
   | /    \                /       |                 \                    \ \
   |/       \  Filament B/         |                  \  Tight Crop        \ \
   +-------------------------------+                   +-----------------------+
```

### 2.2 Catastrophic Overlap & Intersection Failure Modes
When two filaments lie in close proximity or physically intersect ($d < L$), their horizontal bounding boxes $B_{\text{HBB}}^A$ and $B_{\text{HBB}}^B$ exhibit massive spatial overlap ($\text{IoU}_{\text{box}} \in [0.50, 0.80]$). Downstream segmentation heads encounter two catastrophic failure patterns:

1. **Over-Merging Cascades**: The binary segmentation head classifies all dark foreground plasma in the RoI crop as belonging to a single object, outputting a composite mask $\hat{y}_{A+B}$. Spatial overlaps $\text{IoU}(y_A, \hat{y}_{A+B})$ and $\text{IoU}(y_B, \hat{y}_{A+B})$ drop below $0.5$, converting potential True Positives into **2 $FN$s and 1 $FP$**.
2. **Boundary Truncation & Fragmentation**: Non-Maximum Suppression (NMS) or feature suppression severs the intersecting region of Filament B, creating fragmented "island" masks that trigger extensive PQ penalties.
3. **Topological Spine Disconnections**: Filaments display spatial variations in optical depth where intermediate segments appear faint. Standard pixel-wise loss functions (Cross-Entropy/Dice) evaluate error uniformly. Because thin connecting spines occupy $<2\%$ of the RoI crop area, standard losses permit the network to sever faint connections without incurring major loss penalties.

---

## 3. Comprehensive Survey of Computer Vision Paradigms

| Paradigm | Overlap Resolution | Barb Preservation | Topological Continuity | Memory Overhead ($2048 \times 2048$) | Benchmark Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Paradigm 1: Axis-Aligned Detectors**<br>*(Mask R-CNN, YOLOv8-Seg)* | Poor (High RoI clutter & bleed) | Moderate | Moderate | Low to Moderate | Sub-optimal |
| **Paradigm 2: Oriented Bounding Box (OBB)**<br>*(Oriented R-CNN, Oriented Mask R-CNN)* | **Excellent** (Spatial isolation) | Good | Good | Moderate | **High** |
| **Paradigm 3: End-to-End Mask Transformers**<br>*(Mask2Former, Panoptic-PartFormer)* | **Excellent** (Bipartite matching) | Moderate (Boundary smoothing) | High | Very High | Moderate to High |
| **Paradigm 4: Prompted Foundation Models**<br>*(SAM, HQ-SAM)* | Dependent on Prompt Quality | **Superior** (High-res token) | Variable | High | **High** *(paired with OBB)* |

---

## 4. Proposed Architecture: The Oriented-Prompted Topological Network (OPT-HQ Net)

To resolve overlap collisions while preserving fine structural details and topological continuity, we introduce **OPT-HQ Net**.

```
                         [ Input Image: 2048x2048 ]
                                     │
                                     ▼
                    [ Swin-L / ConvNeXt-L + FPN Backbone ]
                                     │
                                     ├──► P2 (1/4 scale - High Res Details)
                                     └──► P5 (1/32 scale - Global Context)
                                     │
                                     ▼
                      [ Oriented RPN (Midpoint Offset) ]
                                     │
                                     ▼
                       [ Rotated RoIAlign Feature Crop ]
                                     │
                                     ▼
               [ HQ-SAM Mask Decoder + P2 Feature Fusion Token ]
                                     │
                                     ▼
                     [ Skeleton-Recall Topological Loss ]
```

### 4.1 Pipeline Architecture Narrative

1. **Multi-Scale Feature Extraction Backbone**: Processes $2048 \times 2048 \times 3$ input via Swin-Large or ConvNeXt-Large paired with a Feature Pyramid Network (FPN), generating feature maps $\{P_2, P_3, P_4, P_5\}$. $P_2$ retains high spatial resolution ($512 \times 512$ at $1/4$ scale) while $P_5$ encodes global solar context ($1/32$ scale).
2. **Oriented Region Proposal Network (Oriented RPN)**: Regresses midpoint offset representations $(\Delta x, \Delta y, \Delta w, \Delta h, \Delta \alpha, \Delta \beta)$ to generate rotated proposals $v = (x_c, y_c, w, h, \theta)$ without angular boundary jump discontinuities.
3. **Rotated RoIAlign & Prompt Encoding**: Extracts a fixed-size $28 \times 28$ feature crop using bilinear sampling along the rotated coordinate frame, physically cropping out adjacent overlapping filaments.
4. **HQ-SAM High-Resolution Mask Decoder**: Injects early $P_2$ feature maps via the HQ token directly into the mask decoder, enabling sub-pixel retrieval for thin barbs.
5. **Topologically-Guided Multi-Task Execution**: Optimizes a compound loss function incorporating classification, oriented box regression, focal loss, Dice loss, and Skeleton-Recall Loss.

### 4.2 Mathematical Formulation of Skeleton-Recall Loss

Let $y \in \{0, 1\}^{H \times W}$ denote the binary ground-truth mask, and $\hat{y} \in [0, 1]^{H \times W}$ denote predicted probabilities. The 1D topological skeleton $K(y)$ is extracted via a morphological thinning transform on $y$.

The **Skeleton-Recall Loss** ($\mathcal{L}_{\text{Skeleton}}$) calculates prediction recall specifically along the skeleton pixels:

$$\mathcal{L}_{\text{Skeleton}}(y, \hat{y}) = 1 - \frac{\sum_{i \in K(y)} y_i \cdot \hat{y}_i}{\sum_{i \in K(y)} y_i + \epsilon}$$

The overall multi-task loss is defined as:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Oriented-Box}} + \lambda_2 \mathcal{L}_{\text{Focal}} + \lambda_3 \mathcal{L}_{\text{Dice}} + \lambda_4 \mathcal{L}_{\text{Skeleton}}$$

> [!TIP]
> Setting $\lambda_4 = 1.5$ strongly penalizes zero-probability predictions along the 1D filament skeleton, forcing the network to preserve topological connectivity across faint plasma regions.

### 4.3 Module Architectural Summary

| Module Name | Technical Architecture / Operation | Functional Role in OPT-HQ Net |
| :--- | :--- | :--- |
| **Multi-Scale FPN Backbone** | Swin-Large / ConvNeXt-Large + FPN | Retains high-resolution spatial details at $P_2$ & global context at $P_5$ |
| **Oriented RPN** | Midpoint offset regression for rotated proposals $(x_c, y_c, w, h, \theta)$ | Tight spatial containment; eliminates background clutter & overlap collisions |
| **Rotated RoIAlign** | Bilinear sampling inside rotated coordinates | Physically decouples adjacent overlapping filaments during feature cropping |
| **HQ-SAM Decoder** | High-Quality token fusing early $P_2$ feature maps | Delivers sub-pixel precision for complex barb morphology |
| **Skeleton-Recall Head** | Morphological thinning transform on spine | Eliminates fragmentation artifacts ("islands") to maximize Panoptic Quality |

---

## 5. Implementation Strategy, Training Protocols & Post-Processing Pipeline

### 5.1 Data Pre-Processing & Dynamic Augmentation
* **Dynamic Range Normalization**: Converts 14-bit FITS values ($0$ to $16,384$) using Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast near the solar limb.
* **Limb Masking**: Circular masking sets space background pixels strictly to zero to prevent false positives outside the solar disk.
* **Dynamic Augmentations**: Random full-degree rotations ($0^\circ-360^\circ$), horizontal/vertical flips, scale jittering ($0.8\times-1.2\times$), and multi-scale cropping ($1024 \times 1024$ to $2048 \times 2048$).

### 5.2 Model Training Setup
* **Optimizer**: AdamW with initial learning rate $\eta = 1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, and Cosine Annealing schedule.
* **Hardware & Precision**: Distributed training across $4 \times$ NVIDIA A100 GPUs (80GB VRAM) using Automatic Mixed Precision (AMP / FP16), effective batch size $= 8$.
* **Loss Weights**: $\lambda_1 = 1.0$ (Oriented Box), $\lambda_2 = 2.0$ (Focal Loss), $\lambda_3 = 2.0$ (Dice Loss), $\lambda_4 = 1.5$ (Skeleton-Recall Loss).

### 5.3 Post-Processing & Submission Pipeline

1. **Polygon-Based Rotated NMS**: Calculates exact polygon intersection ($\text{IoU}_{\text{rotated}} > 0.40$) between oriented boxes, eliminating redundant proposals on crossing filaments.
2. **Morphological Cleaning & Island Filtering**: Filters disconnected masks $< 50$ pixels without an independent box proposal; applies $3 \times 3$ morphological closing to bridge faint boundary gaps.
3. **Fortran-Ordered RLE Encoding**: Encodes $2048 \times 2048$ binary masks into RLE strings in column-major order via `pycocotools.mask.encode(np.asfortranarray(mask, dtype=np.uint8))`. Assigns unique keys with suffix indexes (e.g., `20150125172714Mh_1`, `20150125172714Mh_2`).

### 5.4 End-to-End Pipeline Summary

| Pipeline Stage | Processing Operation | Technical Implementation | Metric Target |
| :--- | :--- | :--- | :--- |
| **Pre-Processing** | Dynamic range normalization & limb masking | CLAHE contrast adjustment & circular disk mask | Suppresses atmospheric background noise ($FP$) |
| **Model Inference** | Single-pass forward execution | Swin-L FPN + Oriented RPN + HQ-SAM Decoder | Generates spatially isolated oriented instance masks |
| **Box Suppression** | Rotated Non-Maximum Suppression | Polygon intersection ($\text{IoU}_{\text{rotated}} > 0.40$) | Prevents duplicate box proposals on crossing filaments |
| **Mask Refinement** | Morphological cleaning & area thresholding | $3 \times 3$ closing kernel; area filter $< 50$ px | Eliminates small noise artifacts & connects spines |
| **RLE Assembly** | Column-major string conversion | `pycocotools` Fortran-ordered encoding | Formats predictions for Panoptic Quality evaluation |

---

## 6. Strategic Conclusions & Recommendations

1. **Oriented Spatial Isolation**: Replacing horizontal boxes with Oriented Region Proposals aligns crops tightly along natural filament orientations, eliminating background noise and preventing overlap collisions in dense active regions.
2. **Sub-Pixel Structural Preservation**: Incorporating HQ-SAM's High-Quality Output Token fuses early high-resolution feature maps ($P_2$), maintaining the fine-scale details required to segment filament barbs.
3. **Topologically-Guided Continuity**: Enforcing Skeleton-Recall Loss during training eliminates spine disconnections, mitigating fragmentation penalties under Panoptic Quality ($PQ$).
4. **Metric Alignment**: Combining polygon-based Rotated NMS with Fortran-ordered RLE encoding ensures seamless compliance with MAGFiLO benchmark evaluation rules.