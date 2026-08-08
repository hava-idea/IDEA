# Dataset Setup

## UCMerced LandUse Dataset

**21 classes, 100 images per class (2100 total), 256×256 pixels, GSD ≈ 0.3 m**

1. Download from the official source:
   ```
   http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
   ```
   or mirror: `https://drive.google.com/file/d/1MiKFdQjmJtFGS8sfhq-K3FFyO_nt7XRi`

2. Extract so the layout matches:
   ```
   UCMerced_LandUse/
     Images/
       agricultural/
         agricultural00.tif  agricultural01.tif  ...
       airplane/
       baseballdiamond/
       beach/
       buildings/
       chaparral/
       denseresidential/
       forest/
       freeway/
       golfcourse/
       harbor/
       intersection/
       mediumresidential/
       mobilehomepark/
       overpass/
       parkinglot/
       river/
       runway/
       sparseresidential/
       storagetanks/
       tenniscourt/
   ```

3. Pass the root path (`UCMerced_LandUse/`) to `--data-root`.

---

## RSSCN7 Dataset

**7 classes, 400 images per class (2800 total), 400×400 pixels**

1. Download from GitHub:
   ```
   https://github.com/palewithout/RSSCN7
   ```
   or direct link:
   ```
   https://github.com/palewithout/RSSCN7/archive/refs/heads/master.zip
   ```

2. Expected layout after extraction:
   ```
   RSSCN7/
     aGricultural/
       Scene_aGricultural_00001.jpg  ...
     fForest/
     gGolf/
     mMeadow/
     pParking/
     rResidential/
     sSea/
   ```

3. Pass the root path (`RSSCN7/`) to `--data-root`.

---

## Split protocol

Both datasets use a seeded stratified shuffle split (seed=42 by default):

| Split | Ratio | Role |
|-------|-------|------|
| train | 70 %  | Candidate pool for FAISS indexing and PCMA fitting |
| val   | 15 %  | Zero-shot inference for prior estimation (C_hard, H) |
| test  | 15 %  | Query set — final accuracy reported in Table 2 |

**The seed must not be changed** to reproduce the exact Table 2 numbers. The
seed is stored in every results JSON under `"seed"` for verification.

> **Important:** The validation split is the only split used for prior
> estimation. Test data never contributes to `C_hard` or `H`.
