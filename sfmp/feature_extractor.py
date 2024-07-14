import os
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image


class FeatureExtractor:
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "artifacts", "feature_extractor.onnx"
        ),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
        image_size: List[int] = [224, 224],
    ):
        self.image_size = tuple(image_size)
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )

    def prepare_input(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.image_size)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = np.array(image)
        image = image / 255.0
        image = image - mean
        image = image / std
        image = np.moveaxis(image, -1, 0)

        return image

    def inference_batch(self, image_list: List[Image.Image]) -> np.array:
        preprocessed_images = np.c_[list(map(self.prepare_input, image_list))]

        features = self.session.run(
            None, {"input": preprocessed_images.astype(np.float16)}
        )[0]

        return features

    def inference(self, image: Image.Image) -> np.ndarray:
        return self.inference_batch([image])
