# Third-party software, models and data

**This repository redistributes no model weights, no slides and no per-tile or
per-cell data derived from slides** (`smoke/slides.json` holds a few per-slide
summary counts). Everything below is obtained by the user from its own source,
under that source's terms.

How the licence information was obtained:

* Python packages: installed package metadata, read with `pip-licenses
  --from=mixed` on 2026-10-16 in the environment recorded in
  `requirements-lock-linux.txt`. Metadata is what each package declares; it
  was not checked against the packages' licence files.
* Model weights: the `license` field the Hugging Face API returned for each
  repository on 2026-10-16 (`https://huggingface.co/api/models/<repo>`). That is
  the repository owner's tag, not the model card or the access terms, which
  were **not** read.
* CellViT: the PyPI metadata of the `cellvit` package on 2026-10-16.
* UBC-OCEAN: not checked.

## Direct Python dependencies

| Package | Version installed | Licence (package metadata) | Used for |
|---|---|---|---|
| numpy | 1.26.4 | BSD License | everything |
| scipy | 1.17.1 | BSD License | statistics, image filters |
| pandas | 3.0.2 | BSD License | tables |
| scikit-learn | 1.8.0 | BSD-3-Clause | folds, metrics, ridge, controls |
| scikit-image | 0.26.0 | BSD License | texture features |
| matplotlib | 3.10.8 | Python Software Foundation License | figures |
| Pillow | 10.4.0 | Historical Permission Notice and Disclaimer (HPND) | images |
| opencv-python-headless | 4.7.0.72 | Apache Software License | contour shape features |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | progress bars |
| torch | 2.5.1+cu121 | BSD License | models |
| torchvision | 0.20.1+cu121 | BSD | required by timm |
| openslide-python | 1.4.3 | LGPL-2.1-only AND BSD-3-Clause AND MIT AND LicenseRef-Public-Domain | slide reading |
| tiffslide | 2.5.1 | BSD License | slide reading |
| pyvips | 3.1.1 | MIT License | slide thumbnails, PNG to TIFF conversion |
| tifffile | 2025.5.10 | BSD-3-Clause | TIFF access (via tiffslide) |
| zarr | 2.18.7 | MIT License | TIFF access (via tiffslide) |
| timm | 1.0.29 | Apache Software License | UNI2-h, H-optimus-0 |
| transformers | 5.16.1 | Apache 2.0 License | Phikon, Phikon-v2, Hibou-L |
| huggingface_hub | 1.33.0 | Apache Software License | model download |
| safetensors | 0.8.0 | Apache Software License | model weights |
| einops | 0.8.2 | MIT License | listed by `check_fm_env.py` |

System libraries, installed with the operating system's package manager:
libvips and OpenSlide. Their licences were not checked here: UNVERIFIED.

## Models

| Model | Repository | HF licence tag | Access | Notes |
|---|---|---|---|---|
| Phikon-v2 | `owkin/phikon-v2` | `other` | ungated | Terms not read: UNVERIFIED, read the model card |
| Phikon | `owkin/phikon` | `other` | ungated | Terms not read: UNVERIFIED, read the model card |
| UNI2-h | `MahmoodLab/UNI2-h` | `cc-by-nc-nd-4.0` | gated (accept terms, `HF_TOKEN`) | Non-commercial, no derivatives per the tag; access terms UNVERIFIED |
| H-optimus-0 | `bioptimus/H-optimus-0` | `apache-2.0` | gated (accept terms, `HF_TOKEN`) | Access terms UNVERIFIED |
| Hibou-L | `histai/hibou-L` | `apache-2.0` | gated (accept terms, `HF_TOKEN`) | Access terms UNVERIFIED. Loaded with stock transformers classes, not the repository's remote code |
| CellViT-SAM-H (`CellViT-SAM-H-x40-AMP.pth`) | downloaded by the `cellvit` package | not checked | | Checkpoint licence UNVERIFIED |

The `cellvit` package (1.0.9 on PyPI) declares "Apache 2.0 with common clause,
see LICENSE" and links `https://github.com/TIO-IKIM/CellViT-Inference/blob/main/LICENSE`.
A Commons Clause restricts selling the software; read the licence before any
commercial use. The version used for the original run was not recorded.

## Data

| Dataset | Source | Terms |
|---|---|---|
| UBC-OCEAN | Kaggle competition `UBC-OCEAN` | UNVERIFIED: read the competition rules; accepting them is required before download |
| Ben Taeb et al. external corpus (80 slides) | the original publication's data release | UNVERIFIED |

No slide, crop, feature file or embedding from either dataset is included here.

## Full environment (all installed packages, including dev tools)

| Name                     | Version      | License                                                             | URL                                                        |
|--------------------------|--------------|---------------------------------------------------------------------|------------------------------------------------------------|
| Deprecated               | 3.0.0        | MIT                                                                 | https://github.com/laurent-laporte-pro/deprecated          |
| ImageIO                  | 2.38.1       | BSD-2-Clause                                                        | https://github.com/imageio/imageio                         |
| Jinja2                   | 3.1.6        | BSD License                                                         | https://github.com/pallets/jinja/                          |
| MarkupSafe               | 3.0.4        | BSD-3-Clause                                                        | https://github.com/pallets/markupsafe/                     |
| PyYAML                   | 6.0.3        | MIT License                                                         | https://pyyaml.org/                                        |
| Pygments                 | 2.21.0       | BSD-2-Clause                                                        | https://pygments.org                                       |
| annotated-doc            | 0.0.5        | MIT                                                                 | https://github.com/fastapi/annotated-doc                   |
| anyio                    | 4.15.1       | MIT                                                                 | https://anyio.readthedocs.io/en/stable/versionhistory.html |
| asciitree                | 0.3.3        | MIT                                                                 | http://github.com/mbr/asciitree                            |
| certifi                  | 2026.7.22    | Mozilla Public License 2.0 (MPL 2.0)                                | https://github.com/certifi/python-certifi                  |
| cffi                     | 2.1.1        | MIT-0                                                               | https://cffi.readthedocs.io/en/latest/whatsnew.html        |
| cfgv                     | 3.5.0        | MIT                                                                 | https://github.com/asottile/cfgv                           |
| click                    | 8.5.0        | BSD-3-Clause                                                        | https://github.com/pallets/click/                          |
| cloudpickle              | 3.1.2        | BSD License                                                         | https://github.com/cloudpipe/cloudpickle                   |
| contourpy                | 1.3.3        | BSD License                                                         | https://github.com/contourpy/contourpy                     |
| cycler                   | 0.12.1       | BSD License                                                         | https://matplotlib.org/cycler/                             |
| distlib                  | 0.4.3        | Python Software Foundation License                                  | https://github.com/pypa/distlib                            |
| einops                   | 0.8.2        | MIT License                                                         | https://github.com/arogozhnikov/einops                     |
| fasteners                | 0.20         | Apache-2.0                                                          | https://github.com/harlowja/fasteners                      |
| filelock                 | 4.2.0        | MIT                                                                 | https://github.com/tox-dev/py-filelock                     |
| fonttools                | 4.67.0       | MIT                                                                 | http://github.com/fonttools/fonttools                      |
| fsspec                   | 2026.9.0     | BSD-3-Clause                                                        | https://github.com/fsspec/filesystem_spec                  |
| h11                      | 0.16.0       | MIT License                                                         | https://github.com/python-hyper/h11                        |
| hf-xet                   | 1.7.0        | Apache-2.0                                                          | https://github.com/huggingface/xet-core                    |
| httpcore                 | 1.0.9        | BSD-3-Clause                                                        | https://www.encode.io/httpcore/                            |
| httpx                    | 0.28.1       | BSD License                                                         | https://github.com/encode/httpx                            |
| huggingface_hub          | 1.33.0       | Apache Software License                                             | https://github.com/huggingface/huggingface_hub             |
| identify                 | 2.6.20       | MIT                                                                 | https://github.com/pre-commit/identify                     |
| idna                     | 3.20         | BSD-3-Clause                                                        | https://github.com/kjd/idna                                |
| imagecodecs              | 2026.1.14    | BSD-3-Clause                                                        | https://www.cgohlke.com                                    |
| iniconfig                | 2.3.1        | MIT                                                                 | https://github.com/pytest-dev/iniconfig                    |
| joblib                   | 1.6.0        | BSD-3-Clause                                                        | https://joblib.readthedocs.io                              |
| kiwisolver               | 1.5.1        | BSD License                                                         | https://github.com/nucleic/kiwi                            |
| lazy-loader              | 0.6          | BSD-3-Clause                                                        | https://github.com/scientific-python/lazy-loader           |
| markdown-it-py           | 4.2.0        | MIT License                                                         | https://github.com/executablebooks/markdown-it-py          |
| matplotlib               | 3.10.8       | Python Software Foundation License                                  | https://matplotlib.org                                     |
| mdurl                    | 0.1.2        | MIT License                                                         | https://github.com/executablebooks/mdurl                   |
| mpmath                   | 1.3.0        | BSD License                                                         | http://mpmath.org/                                         |
| networkx                 | 3.7          | BSD-3-Clause                                                        | https://networkx.org/                                      |
| nodeenv                  | 1.11.0       | BSD License                                                         | https://github.com/ekalinin/nodeenv                        |
| numcodecs                | 0.15.1       | MIT License                                                         | https://github.com/zarr-developers/numcodecs               |
| numpy                    | 1.26.4       | BSD License                                                         | https://numpy.org                                          |
| nvidia-cublas-cu12       | 12.1.3.1     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cuda-cupti-cu12   | 12.1.105     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cuda-nvrtc-cu12   | 12.1.105     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cuda-runtime-cu12 | 12.1.105     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cudnn-cu12        | 9.1.0.70     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cufft-cu12        | 11.0.2.54    | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-curand-cu12       | 10.3.2.106   | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cusolver-cu12     | 11.4.5.107   | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-cusparse-cu12     | 12.1.0.106   | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-nccl-cu12         | 2.21.5       | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-nvjitlink-cu12    | 12.9.86      | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| nvidia-nvtx-cu12         | 12.1.105     | Other/Proprietary License                                           | https://developer.nvidia.com/cuda-zone                     |
| opencv-python-headless   | 4.7.0.72     | Apache Software License                                             | https://github.com/opencv/opencv-python                    |
| openslide-python         | 1.4.3        | LGPL-2.1-only AND BSD-3-Clause AND MIT AND LicenseRef-Public-Domain | https://openslide.org/                                     |
| packaging                | 26.3         | Apache-2.0 OR BSD-2-Clause                                          | https://github.com/pypa/packaging                          |
| pandas                   | 3.0.2        | BSD License                                                         | https://pandas.pydata.org                                  |
| pillow                   | 10.4.0       | Historical Permission Notice and Disclaimer (HPND)                  | https://python-pillow.org                                  |
| platformdirs             | 4.13.3       | MIT                                                                 | https://github.com/tox-dev/platformdirs                    |
| pluggy                   | 1.7.0        | MIT License                                                         | https://github.com/pytest-dev/pluggy                       |
| pre_commit               | 4.7.0        | MIT                                                                 | https://github.com/pre-commit/pre-commit                   |
| pycparser                | 3.11         | BSD-3-Clause                                                        | https://github.com/eliben/pycparser                        |
| pyparsing                | 3.3.3        | MIT                                                                 | https://github.com/pyparsing/pyparsing/                    |
| pytest                   | 9.1.1        | MIT                                                                 | https://docs.pytest.org/en/latest/                         |
| python-dateutil          | 2.9.0.post0  | Apache Software License; BSD License                                | https://github.com/dateutil/dateutil                       |
| python-discovery         | 1.6.3        | MIT License                                                         | https://github.com/tox-dev/python-discovery                |
| pyvips                   | 3.1.1        | MIT License                                                         | https://github.com/libvips/pyvips                          |
| regex                    | 2026.9.29    | Apache-2.0 AND CNRI-Python                                          | https://github.com/mrabarnett/mrab-regex                   |
| rich                     | 15.0.0       | MIT License                                                         | https://github.com/Textualize/rich                         |
| ruff                     | 0.17.1       | MIT                                                                 | https://docs.astral.sh/ruff                                |
| safetensors              | 0.8.0        | Apache Software License                                             | https://github.com/huggingface/safetensors                 |
| scikit-image             | 0.26.0       | BSD License                                                         | https://scikit-image.org                                   |
| scikit-learn             | 1.8.0        | BSD-3-Clause                                                        | https://scikit-learn.org                                   |
| scipy                    | 1.17.1       | BSD License                                                         | https://scipy.org/                                         |
| shellingham              | 1.5.4        | ISC License (ISCL)                                                  | https://github.com/sarugaku/shellingham                    |
| six                      | 1.17.0       | MIT License                                                         | https://github.com/benjaminp/six                           |
| sympy                    | 1.13.1       | BSD License                                                         | https://sympy.org                                          |
| threadpoolctl            | 3.7.0        | BSD-3-Clause                                                        | https://github.com/joblib/threadpoolctl                    |
| tifffile                 | 2025.5.10    | BSD-3-Clause                                                        | https://www.cgohlke.com                                    |
| tiffslide                | 2.5.1        | BSD License                                                         | https://github.com/Bayer-Group/tiffslide                   |
| timm                     | 1.0.29       | Apache Software License                                             | https://github.com/huggingface/pytorch-image-models        |
| tokenizers               | 0.23.3       | Apache Software License                                             | https://github.com/huggingface/tokenizers                  |
| torch                    | 2.5.1+cu121  | BSD License                                                         | https://pytorch.org/                                       |
| torchvision              | 0.20.1+cu121 | BSD                                                                 | https://github.com/pytorch/vision                          |
| tqdm                     | 4.70.1       | MPL-2.0 AND MIT                                                     | https://tqdm.github.io                                     |
| transformers             | 5.16.1       | Apache 2.0 License                                                  | https://github.com/huggingface/transformers                |
| triton                   | 3.1.0        | MIT License                                                         | https://github.com/triton-lang/triton/                     |
| typer                    | 0.27.3       | MIT                                                                 | https://github.com/fastapi/typer                           |
| typing_extensions        | 4.16.0       | PSF-2.0                                                             | https://github.com/python/typing_extensions                |
| virtualenv               | 21.14.8      | MIT                                                                 | https://github.com/pypa/virtualenv                         |
| wrapt                    | 2.5.1        | BSD-2-Clause                                                        | https://github.com/GrahamDumpleton/wrapt                   |
| zarr                     | 2.18.7       | MIT License                                                         | https://github.com/zarr-developers/zarr-python             |
