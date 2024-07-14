import argparse
import os
import shutil
from glob import glob

from sfmp import Sfmp


def get_args():
    parser = argparse.ArgumentParser(description="Separate for me please.")

    parser.add_argument(
        "images_glob",
        type=str,
        nargs="+",
        help='Glob pattern(s) for the images to be processed (e.g., "images/*.jpg").',
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "artifacts", "feature_extractor.onnx"
        ),
        help="Path to the ONNX feature extractor (default: artifacts/feature_extractor.onnx).",
    )

    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Execution provider for ONNX Runtime (default: CPUExecutionProvider).",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Size to which images should be resized (default: (224, 224)).",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=2,
        help="Number of clusters for clustering algorithm (default: 2).",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.getcwd(), "sfmp"),
        help="Path to the output folder (default: ./sfmp).",
    )

    return parser.parse_args()


def main():
    args = get_args()

    sfmp = Sfmp(
        model_path=args.model_path,
        providers=args.providers,
        image_size=args.image_size,
    )

    image_list = sum(
        list(map(lambda image_glob: glob(image_glob), args.images_glob)), []
    )

    os.makedirs(args.output_path, exist_ok=True)

    clusters = sfmp.separate(image_list, n_clusters=args.n_clusters)
    for cluster, cluster_image_list in clusters.items():
        cluster_dirname = f"cluster_{str(cluster).zfill(4)}"
        cluster_path = os.path.join(args.output_path, cluster_dirname)
        os.makedirs(cluster_path, exist_ok=True)
        for cluster_image_path in cluster_image_list:
            shutil.copy(cluster_image_path, cluster_path)


if __name__ == "__main__":
    main()
