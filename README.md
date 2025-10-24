# torch-power-weighted-projection

[![License](https://img.shields.io/pypi/l/torch-power-weighted-projection.svg?color=green)](https://github.com/jojoelfe/torch-power-weighted-projection/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-power-weighted-projection.svg?color=green)](https://pypi.org/project/torch-power-weighted-projection)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-power-weighted-projection.svg?color=green)](https://python.org)
[![CI](https://github.com/jojoelfe/torch-power-weighted-projection/actions/workflows/ci.yml/badge.svg)](https://github.com/jojoelfe/torch-power-weighted-projection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jojoelfe/torch-power-weighted-projection/branch/main/graph/badge.svg)](https://codecov.io/gh/jojoelfe/torch-power-weighted-projection)

Create power weighted projections of volumes using GPU-accelerated ray casting with PyTorch.

## Installation

```sh
pip install torch-power-weighted-projection
```

## Features

- **GPU-accelerated ray casting** using PyTorch for fast projection generation
- **Power-weighted projection** with configurable power parameter for feature enhancement
- **Arbitrary viewing angles** using ZYZ Euler angle rotations
- **Proper boundary handling** with out-of-bounds masking to avoid artifacts
- **Memory-efficient chunked processing** for large volumes
- **Flexible output sizing** with automatic dimension calculation

## Usage

### Basic Example

```python
import numpy as np
from torch_power_weighted_projection import power_projection_raycast_gpu

# Load or create your 3D volume data
volume_data = np.random.randn(100, 100, 100).astype(np.float32)

# Generate a projection with specified Euler angles
projection = power_projection_raycast_gpu(
    data=volume_data,
    euler_angles=(0, 0, 0),  # (alpha, beta, gamma) in radians
    p=-5,                     # Power parameter (negative values emphasize edges)
    num_samples=800,          # Number of samples along each ray
    device='cuda',            # Use 'cpu' if CUDA is not available
    output_size=(800, 800),   # Output image dimensions
    chunk_size=32             # Process 32 rows at a time to manage memory
)

# projection is a 2D numpy array that can be visualized with matplotlib
import matplotlib.pyplot as plt
plt.imshow(projection, cmap='gray')
plt.show()
```

### Creating Tilt Series

```python
import matplotlib.animation as manimation

# Create an animation by varying the tilt angle
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=15)

fig = plt.figure(figsize=(12, 8))

with writer.saving(fig, "tilt_series.mp4", 100):
    for i in range(100):
        # Rotate around Y-axis (beta angle)
        projection = power_projection_raycast_gpu(
            volume_data,
            euler_angles=(0, i * 0.005, 0),
            p=-5,
            num_samples=800,
            chunk_size=4,
            output_size=(1200, 800)
        )
        plt.clf()
        plt.imshow(projection, cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
        writer.grab_frame()
```

### Working with Cryo-EM Data

```python
import mrcfile

# Load an MRC/MAP file
with mrcfile.open('volume.mrc') as mrc:
    volume_data = mrc.data.astype(np.float32)

# Normalize the data
volume_data -= volume_data.mean()
volume_data /= volume_data.std()

# Generate projection
projection = power_projection_raycast_gpu(
    volume_data,
    euler_angles=(0, 0.5, 0),
    p=-5,
    num_samples=None,  # Auto-calculate based on volume diagonal
    device='cuda'
)
```

## API Reference

### `power_projection_raycast_gpu(data, euler_angles, num_samples=None, p=-5, device='cuda', output_size=None, chunk_size=32)`

Ray casting projection with power weighting and proper masking of out-of-bounds samples.

**Parameters:**
- `data` (np.ndarray or torch.Tensor): 3D volume data
- `euler_angles` (tuple): (alpha, beta, gamma) ZYZ Euler angles in radians
- `num_samples` (int, optional): Number of samples along each ray. Default: volume diagonal length
- `p` (float): Power parameter for weighted projection. Default: -5 (negative values emphasize edges)
- `device` (str): Device to use for computation ('cuda' or 'cpu'). Default: 'cuda'
- `output_size` (tuple, optional): (height, width) for output image. Default: auto-calculated
- `chunk_size` (int): Number of rows to process at once for memory management. Default: 32

**Returns:**
- `np.ndarray`: 2D projection of the volume

### `zyz_to_rotation_matrix(alpha, beta, gamma, device='cuda')`

Convert ZYZ Euler angles to a 3x3 rotation matrix.

**Parameters:**
- `alpha` (float): First rotation angle around Z-axis (in radians)
- `beta` (float): Second rotation angle around Y-axis (in radians)
- `gamma` (float): Third rotation angle around Z-axis (in radians)
- `device` (str): Device to use for computation. Default: 'cuda'

**Returns:**
- `torch.Tensor`: 3x3 rotation matrix

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jojoelfe/torch-power-weighted-projection --clone
# or just
# gh repo clone jojoelfe/torch-power-weighted-projection
cd torch-power-weighted-projection
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
