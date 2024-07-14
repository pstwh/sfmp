import os
import unittest
from glob import glob

from sfmp import Sfmp


class TestSfmp(unittest.TestCase):
    def setUp(self):
        self.sfmp = Sfmp()

    def test_sfmp(self):
        sfmp_examples_glob = os.path.join(
            os.path.dirname(__file__), "../examples/*.jpg"
        )
        sfmp_examples = glob(sfmp_examples_glob)

        clusters = self.sfmp.separate(sfmp_examples, n_clusters=2)

        check = []
        for _, cluster_image_list in clusters.items():
            classes = list(
                set(
                    map(
                        lambda p: os.path.splitext(os.path.basename(p))[0].split(".")[
                            0
                        ],
                        cluster_image_list,
                    )
                )
            )
            check.append(len(classes) == 1)

        self.assertTrue(all(check))


if __name__ == "__main__":
    unittest.main()
