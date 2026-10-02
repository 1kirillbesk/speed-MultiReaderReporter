import unittest
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "speed_MultiReaderReporter"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from speed_MultiReaderReporter.exported_core_functions import get_exported_augmentation_methods_module
from speed_MultiReaderReporter.lfp_prediction_common import (
    LFPInterpolationModule,
    _build_feature_log2_helper_prediction_scales,
)


class FeatureLog2HelperScaleTests(unittest.TestCase):
    def test_helper_prediction_multiplier_sets_feature_log2_scale_cap(self):
        exported_aug = get_exported_augmentation_methods_module()
        module = LFPInterpolationModule(
            input_steps=6,
            augmentation_reference_soh=0.92,
        )
        train_targets = np.asarray([80.0, 100.0], dtype=float)
        predicted_test_targets = np.asarray([120.0, 140.0], dtype=float)

        scales, stats = _build_feature_log2_helper_prediction_scales(
            exported_aug,
            module,
            train_targets,
            predicted_test_targets,
        )

        self.assertTrue(scales, "expected helper-derived feature_log2 scales")
        self.assertAlmostEqual(2.0, float(stats["feature_log2_helper_prediction_multiplier"]))
        self.assertAlmostEqual(2.0, float(stats["feature_log2_lifetime_prediction_multiplier"]))
        self.assertAlmostEqual(140.0, float(stats["predicted_test_max_cycle_life"]))
        self.assertAlmostEqual(280.0, float(stats["predicted_test_upper_cycle_life"]))
        self.assertAlmostEqual(2.8, float(stats["lifetime_coverage_scale"]))
        self.assertAlmostEqual(2.8, float(stats["feature_log2_lifetime_scale"]))
        self.assertAlmostEqual(2.8, float(stats["max_augmentation_scale"]))
        self.assertAlmostEqual(2.8, float(scales[-1]))


if __name__ == "__main__":
    unittest.main()
