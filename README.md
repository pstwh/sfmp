# sfmp (separate for me please)

Sfmp is a tool to quickly separate images using clustering based on features extracted through a convolutional neural network.
It's really lightweight, using only a mobilenetv2 for feature extraction and kmeans for clustering.

## Installation

To install sfmp, follow there steps:

For cpu
```
pip install "sfmp[cpu]"
```

For cuda 11.X
```
pip install "sfmp[gpu]"
```

For cuda 12.X
```
pip install "sfmp[gpu]" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Usage

### Command Line Interface

You can use sfmp from the command line by providing a list of globs:

```
sfmp ../examples/*.jpg ../examples2/*.jpg
```

If you want to use CUDA:


```
sfmp ../examples/*.jpg ../examples2/*.jpg --provider CUDAExecutionProvider
```

Check all execution providers [here](https://onnxruntime.ai/docs/execution-providers/).

To specify the number of clusters:

```
sfmp ../examples/*.jpg ../examples2/*.jpg --n_clusters 2
```

or output dir

To specify the number of clusters:

```
sfmp ../examples/*.jpg ../examples2/*.jpg --output_path result
```

### Example

Before
```
.
├── cat.1.jpg
├── cat.2.jpg
├── dog.3061.jpg
└── dog.3062.jpg
```

```
sfmp ../examples/*.jpg --n_clusters 2
```

After

```
├── cluster_0000
│   ├── dog.3061.jpg
│   └── dog.3062.jpg
└── cluster_0001
    ├── cat.1.jpg
    └── cat.2.jpg
```

