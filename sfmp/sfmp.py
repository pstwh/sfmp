import concurrent.futures
import os
from typing import List

import numpy as np
import onnxruntime as ort
from sfmp.feature_extractor import FeatureExtractor
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Sfmp:
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "artifacts", "feature_extractor.onnx"
        ),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
        image_size: List[int] = [224, 224],
    ):
        self.feature_extractor = FeatureExtractor(
            model_path, sess_options, providers, image_size
        )

    def extract_features(
        self, image_list: List[str], max_workers: int = os.cpu_count()
    ):
        images_features = {}

        def extract_features_worker(image_path):
            image = Image.open(image_path).convert("RGB")
            return image_path, self.feature_extractor.inference(image).reshape(-1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            output = list(
                tqdm(
                    executor.map(extract_features_worker, image_list),
                    total=len(image_list),
                    desc="Extracting features",
                )
            )

        for image, features in output:
            images_features[image] = features

        return np.c_[list(images_features.values())]

    def fit_transform_clusters(
        self, features: np.ndarray, n_clusters: int, random_state: int = 0
    ):
        kmeans = KMeans(
            n_clusters=n_clusters, n_init="auto", random_state=random_state
        ).fit(StandardScaler().fit_transform(features))

        return kmeans.labels_

    def separate(
        self, image_list: List[str], n_clusters: int = 2, random_state: int = 0
    ):
        features = self.extract_features(image_list)
        labels = self.fit_transform_clusters(
            features, n_clusters, random_state=random_state
        )

        clusters = {}
        for label, image_path in zip(labels, image_list):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(image_path)

        return clusters
