# semanticMoE

semanticMoE is a semantic-guided mixture-of-experts framework for brain-age prediction and individualized brain-age composition profiling from multimodal imaging-derived phenotypes (IDPs).
The model uses semantic embeddings of IDP metadata to guide routing to specialized local experts, while an always-active global expert processes whole-brain and hemispheric measures. Their outputs are combined to predict brain age and form an individual composition profile. Whereas the brain-age gap summarizes overall brain-age deviation as a single scalar, the profile describes how the prediction is composed across different imaging patterns.

## Setup

Install from the directory containing `semanticMoE/`:

```bash
python -m pip install ./semanticMoE
```
To use without installing the package, install its dependencies and run Python from the directory containing `semanticMoE/`:

```bash
python -m pip install "torch>=2.6" "numpy>=1.24" "pandas>=2.0"
```

Then use the same Python example below. If running from another directory, add the folder's parent directory to the Python search path before importing:

```python
import sys
sys.path.insert(0, "/path/to/directory/containing/semanticMoE")
from semanticMoE import model
```

Use absolute paths for the sample and checkpoint when running from another directory.

## Input and output

Input: a numeric NumPy array of shape `(3630,)`, containing one subject's raw IDP values **in the row order of `assets/idp_metadata.csv`** (excluding the header). Units and UK Biobank field IDs are provided in that file. Do not include a subject ID or age. Missing measurements use `np.nan`; at most 20% may be missing. Preprocessing is applied internally.

Output: a Python `float` representing predicted brain age in years.

## Usage

Run from the directory containing `semanticMoE/`:

```python
import numpy as np
from semanticMoE import model

# Synthetic single-subject sample; replace with your own ordered IDP values.
idps = np.load("semanticMoE/examples/sample_idps.npy")

model.model_predict(idps, pre_trained="semanticMoE/checkpoints/brain_age_seed2026.pt")

```

`pre_trained` accepts a bundled checkpoint path (seed 2026, 42, or 3407). Omitting it uses random model parameters without loading trained weights. The interface runs on CPU and accepts one subject per call.

For UK Biobank data, rfMRI bulk fields must be expanded into the individual edge/node values named in the metadata before arranging the input array.
