# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for detectron2.calibration.

Covers: known-distortion correction with before/after numbers, rank-metric
preservation (including ties), per-class operating points vs a uniform
threshold, long-tail shrinkage, save/load consistency, class-set validation,
fallback paths, domain-shift reporting, tiny batches, boundary scores, and
shard-aggregation consistency.
"""

import json
import logging
import unittest

import numpy as np
import torch

from detectron2.calibration import (
    CALIBRATED_SCORE_FIELD,
    MODE_GLOBAL,
    MODE_SHRUNK,
    CalibrationMismatchError,
    CalibrationRangeError,
    ScoreCalibrator,
    average_precision,
    calibrate_instances,
    detect_domain_shift,
    expected_calibration_error,
    fit_calibrator,
    fit_calibrator_from_data,
    gather_calibration_data,
    load_calibration,
    merge_calibration_data,
    save_calibration,
    solve_thresholds_at_fpr,
    try_load_calibration,
    uniform_threshold_at_fpr,
)
from detectron2.structures import Boxes, Instances


def _make_distorted(n, rng, num_classes=3, power=1.0 / 3.0):
    """Scores inflated away from the true posterior: s = q ** power."""
    q = rng.beta(2.0, 2.0, n)
    scores = np.clip(q**power, 0.0, 1.0)
    labels = (rng.random(n) < q).astype(np.float64)
    classes = rng.integers(0, num_classes, n)
    return scores, labels, classes


class TestCalibrationQuality(unittest.TestCase):
    def test_known_distortion_reduces_calibration_error(self):
        rng = np.random.default_rng(7)
        scores, labels, classes = _make_distorted(30000, rng)
        calibrator, report = ScoreCalibrator.fit(scores, labels, classes)

        calibrated = calibrator.predict(scores, classes)
        ece_before = expected_calibration_error(scores, labels)
        ece_after = expected_calibration_error(calibrated, labels)
        print("\n[done-1] known distortion s = q^(1/3):")
        print(f"  ECE    before={ece_before:.4f} after={ece_after:.4f}")
        print(
            f"  Brier  before={report.brier_before:.4f} after={report.brier_after:.4f}"
        )
        self.assertLess(ece_after, ece_before * 0.5)
        self.assertLess(report.brier_after, report.brier_before)
        # Reported numbers must match an independent recomputation.
        self.assertAlmostEqual(report.ece_before, ece_before, places=10)
        self.assertAlmostEqual(report.ece_after, ece_after, places=10)

    def test_single_class_calibration(self):
        rng = np.random.default_rng(11)
        scores, labels, classes = _make_distorted(8000, rng, num_classes=1)
        calibrator, report = ScoreCalibrator.fit(scores, labels, classes)
        self.assertEqual(calibrator.num_classes, 1)
        self.assertLess(report.ece_after, report.ece_before)


class TestRankPreservation(unittest.TestCase):
    def test_rank_metrics_preserved_and_ties_stable(self):
        rng = np.random.default_rng(13)
        scores, labels, classes = _make_distorted(12000, rng)
        # Inject heavy ties, plus exact 0 and 1 scores.
        scores = np.round(scores, 1)
        scores[:50] = 0.0
        scores[50:100] = 1.0
        calibrator, _ = ScoreCalibrator.fit(scores, labels, classes)
        calibrated = calibrator.predict(scores, classes)

        # Strictly increasing within each class: distinct inputs keep their
        # order, ties map to ties.
        for k in range(calibrator.num_classes):
            s_k, p_k = scores[classes == k], calibrated[classes == k]
            order = np.argsort(s_k, kind="stable")
            s_sorted, p_sorted = s_k[order], p_k[order]
            distinct = np.diff(s_sorted) > 0
            self.assertTrue(np.all(np.diff(p_sorted)[distinct] > 0))
            tied = np.diff(s_sorted) == 0
            self.assertTrue(np.all(np.diff(p_sorted)[tied] == 0))

        # Per-class AP (rank-based) must be identical before/after.
        print("\n[done-2] per-class AP before -> after calibration:")
        for k in range(calibrator.num_classes):
            mask = classes == k
            ap_before = average_precision(labels[mask], scores[mask])
            ap_after = average_precision(labels[mask], calibrated[mask])
            print(f"  class {k}: {ap_before:.6f} -> {ap_after:.6f}")
            self.assertAlmostEqual(ap_before, ap_after, places=12)


class TestOperatingPoints(unittest.TestCase):
    def _skewed_data(self, rng):
        """Class 0 negatives score high, class 1 negatives score low; a single
        uniform threshold cannot serve both."""
        n_pos, n_neg = 400, 400
        s0 = np.concatenate([rng.beta(8, 2, n_pos), rng.beta(6, 2, n_neg)])
        s1 = np.concatenate([rng.beta(8, 2, n_pos), rng.beta(1, 8, n_neg)])
        scores = np.concatenate([s0, s1])
        labels = np.array([1.0] * n_pos + [0.0] * n_neg + [1.0] * n_pos + [0.0] * n_neg)
        classes = np.array([0] * (2 * n_pos) + [1] * (2 * n_pos))
        return scores, labels, classes

    def test_per_class_thresholds_beat_uniform(self):
        rng = np.random.default_rng(17)
        scores, labels, classes = self._skewed_data(rng)
        target_fpr = 0.05
        thresholds, report = solve_thresholds_at_fpr(
            scores, labels, classes, num_classes=2, target_fpr=target_fpr
        )
        uniform = uniform_threshold_at_fpr(scores, labels, target_fpr)

        print("\n[done-3] per-class thresholds at target FPR=0.05:")
        det_per_class, det_uniform = [], []
        for k in range(2):
            pos = scores[(classes == k) & (labels > 0.5)]
            neg = scores[(classes == k) & (labels < 0.5)]
            d_pc = float((pos >= thresholds[k]).mean())
            d_un = float((pos >= uniform).mean())
            fpr_un = float((neg >= uniform).mean())
            det_per_class.append(d_pc)
            det_uniform.append(d_un)
            print(
                f"  class {k}: threshold={thresholds[k]:.4f} (uniform {uniform:.4f}), "
                f"detection {d_pc:.4f} vs uniform {d_un:.4f}, "
                f"achieved FPR={report[k].achieved_fpr:.4f} (uniform {fpr_un:.4f})"
            )
            # Every class meets the FPR budget under its own threshold.
            self.assertLessEqual(report[k].achieved_fpr, target_fpr + 0.02)
        # The uniform threshold cannot meet the budget on the high-negative
        # class, and starves the low-negative class of detections.
        neg0 = scores[(classes == 0) & (labels < 0.5)]
        self.assertGreater(float((neg0 >= uniform).mean()), target_fpr)
        self.assertGreater(det_per_class[1], det_uniform[1])
        self.assertGreater(
            sum(det_per_class) / len(det_per_class),
            sum(det_uniform) / len(det_uniform),
        )

    def test_tail_class_uses_shrinkage(self):
        rng = np.random.default_rng(19)
        # Head class: plenty of positives. Tail class: only a handful.
        s_head = rng.beta(2, 2, 4000)
        y_head = (rng.random(4000) < s_head).astype(float)
        s_tail = np.concatenate([rng.beta(5, 2, 5), rng.beta(1, 5, 200)])
        y_tail = np.array([1.0] * 5 + [0.0] * 200)
        scores = np.concatenate([s_head, s_tail])
        labels = np.concatenate([y_head, y_tail])
        classes = np.array([0] * 4000 + [1] * 205)
        calibrator, report = ScoreCalibrator.fit(
            scores, labels, classes, num_classes=2, min_samples=30
        )
        self.assertEqual(report.classes[1].mode, MODE_SHRUNK)
        self.assertIn(1, report.shrunk_classes)
        # The shrunk params must be the exact convex combination of the free
        # tail fit and the global params, with weight n_pos/(n_pos+kappa).
        from detectron2.calibration import fit_platt

        hat = np.array(fit_platt(s_tail, y_tail))
        w = 5 / (5 + calibrator.shrink_strength)
        expected = w * hat + (1 - w) * calibrator.global_params
        np.testing.assert_allclose(calibrator.class_params[1], expected, atol=1e-10)
        # And they must differ from the free fit: shrinkage actually pulled.
        self.assertGreater(np.abs(hat - expected).sum(), 1e-3)

    def test_shrink_strength_configurable(self):
        rng = np.random.default_rng(23)
        s = np.concatenate([rng.beta(2, 2, 3000), rng.beta(5, 2, 8), rng.beta(1, 5, 100)])
        y = np.concatenate(
            [
                (rng.random(3000) < s[:3000]).astype(float),
                np.ones(8),
                np.zeros(100),
            ]
        )
        c = np.array([0] * 3000 + [1] * 108)
        weak, _ = ScoreCalibrator.fit(s, y, c, num_classes=2, shrink_strength=5.0)
        strong, _ = ScoreCalibrator.fit(s, y, c, num_classes=2, shrink_strength=500.0)
        d_weak = np.abs(weak.class_params[1] - weak.global_params).sum()
        d_strong = np.abs(strong.class_params[1] - strong.global_params).sum()
        self.assertLess(d_strong, d_weak)


class TestPersistence(unittest.TestCase):
    def _fit(self):
        rng = np.random.default_rng(29)
        scores, labels, classes = _make_distorted(6000, rng)
        calibrator, _ = ScoreCalibrator.fit(scores, labels, classes)
        return calibrator, scores, classes

    def test_save_load_roundtrip_bit_identical(self):
        import tempfile, os

        calibrator, scores, classes = self._fit()
        class_ids = [f"cls_{k}" for k in range(calibrator.num_classes)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calib.json")
            save_calibration(calibrator, path, class_ids)
            loaded = load_calibration(path, expected_class_ids=class_ids)
            out_before = calibrator.predict(scores, classes)
            out_after = loaded.predict(scores, classes)
            self.assertTrue(np.array_equal(out_before, out_after))
            self.assertEqual(loaded.modes, calibrator.modes)
        print("\n[done-4] save/load roundtrip: outputs bit-identical")

    def test_class_set_mismatch_refused(self):
        import tempfile, os

        calibrator, _, _ = self._fit()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calib.json")
            save_calibration(calibrator, path, ["a", "b", "c"])
            with self.assertRaises(CalibrationMismatchError):
                load_calibration(path, expected_class_ids=["a", "b", "WRONG"])
            with self.assertRaises(CalibrationMismatchError):
                load_calibration(path, expected_class_ids=["a", "b"])
            # Same set, different order, must also be refused (no positional
            # silent alignment).
            with self.assertRaises(CalibrationMismatchError):
                load_calibration(path, expected_class_ids=["b", "a", "c"])

    def test_fallback_paths_are_observable(self):
        import tempfile, os

        calibrator, _, _ = self._fit()
        with tempfile.TemporaryDirectory() as d:
            # Missing file.
            with self.assertLogs("detectron2.calibration.io", level="WARNING") as cm:
                self.assertIsNone(try_load_calibration(os.path.join(d, "nope.json")))
            self.assertIn("uncalibrated", cm.output[0])

            # Version mismatch.
            path = os.path.join(d, "calib.json")
            save_calibration(calibrator, path, ["a", "b", "c"])
            with open(path) as f:
                payload = json.load(f)
            payload["format_version"] = 999
            with open(path, "w") as f:
                json.dump(payload, f)
            with self.assertLogs("detectron2.calibration.io", level="WARNING") as cm:
                self.assertIsNone(try_load_calibration(path, ["a", "b", "c"]))
            self.assertIn("version", cm.output[0])

            # Class count mismatch -> fallback, and the log must not leak names.
            save_calibration(calibrator, path, ["a", "b", "c"])
            logger = logging.getLogger("test.calib")
            with self.assertLogs(logger, level="WARNING") as cm:
                self.assertIsNone(try_load_calibration(path, ["a", "b"], log=logger))
            self.assertNotIn("cls_", cm.output[0])
            self.assertNotIn("'a'", cm.output[0])


class TestFailureModes(unittest.TestCase):
    def test_zero_positive_class_uses_global_and_says_so(self):
        rng = np.random.default_rng(31)
        s0 = rng.beta(2, 2, 3000)
        y0 = (rng.random(3000) < s0).astype(float)
        s1 = rng.beta(2, 2, 500)  # class 1: scores exist, zero positives
        y1 = np.zeros(500)
        scores = np.concatenate([s0, s1])
        labels = np.concatenate([y0, y1])
        classes = np.array([0] * 3000 + [1] * 500)
        calibrator, report = ScoreCalibrator.fit(scores, labels, classes, num_classes=2)
        self.assertEqual(report.classes[1].mode, MODE_GLOBAL)
        self.assertIn(1, report.global_fallback_classes)
        np.testing.assert_array_equal(
            calibrator.class_params[1], calibrator.global_params
        )
        text = report.format()
        self.assertIn("global", text)

    def test_out_of_range_output_raises_with_class_index(self):
        rng = np.random.default_rng(37)
        scores, labels, classes = _make_distorted(4000, rng)
        calibrator, _ = ScoreCalibrator.fit(scores, labels, classes)
        # Corrupt one class's parameters to produce NaN outputs.
        calibrator.class_params[1, 0] = np.nan
        with self.assertRaises(CalibrationRangeError) as ctx:
            calibrator.predict(scores, classes)
        self.assertEqual(ctx.exception.class_indices, [1])
        self.assertIn("1", str(ctx.exception))

    def test_no_annotations_falls_back_without_error(self):
        # No samples at all.
        with self.assertLogs("detectron2.calibration.api", level="INFO") as cm:
            calibrator, report = fit_calibrator_from_data(
                (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
            )
        self.assertIsNone(calibrator)
        self.assertIsNone(report)
        self.assertIn("uncalibrated", cm.output[0])
        # Samples exist but not a single positive (e.g. unlabeled set matched
        # against nothing).
        with self.assertLogs("detectron2.calibration.api", level="INFO"):
            calibrator, _ = fit_calibrator_from_data(
                (
                    np.array([0.5, 0.6]),
                    np.zeros(2),
                    np.zeros(2, dtype=np.int64),
                )
            )
        self.assertIsNone(calibrator)


class TestEdgesAndBatches(unittest.TestCase):
    def test_boundary_scores_and_empty_inputs(self):
        rng = np.random.default_rng(41)
        scores, labels, classes = _make_distorted(4000, rng)
        calibrator, _ = ScoreCalibrator.fit(scores, labels, classes)
        edge = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5])
        cls = np.array([0, 0, 1, 1, 2, 2, 2])
        out = calibrator.predict(edge, cls)
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))
        self.assertLess(out[0], out[1])  # 0 maps below 1
        self.assertEqual(out[4], out[5])  # ties stay tied
        # Empty batch.
        self.assertEqual(calibrator.predict([], []).size, 0)

    def test_single_image_batch_of_one(self):
        rng = np.random.default_rng(43)
        scores, labels, classes = _make_distorted(4000, rng)
        calibrator, _ = ScoreCalibrator.fit(scores, labels, classes)
        out = calibrator.predict(scores[:1], classes[:1])
        self.assertEqual(out.shape, (1,))
        self.assertTrue(0.0 <= out[0] <= 1.0)


class TestShardAggregation(unittest.TestCase):
    def test_unbalanced_shards_aggregate_consistently(self):
        rng = np.random.default_rng(47)
        scores, labels, classes = _make_distorted(9000, rng)
        # Three very unbalanced shards: shard 0 holds nearly all of class 2.
        idx = np.argsort(classes, kind="stable")
        shards = np.array_split(idx, 3)
        triples = [
            (scores[s], labels[s], classes[s]) for s in shards if s.size
        ]
        merged = merge_calibration_data(triples)
        cal_merged, _ = ScoreCalibrator.fit(*merged, num_classes=3)
        cal_full, _ = ScoreCalibrator.fit(scores, labels, classes, num_classes=3)
        np.testing.assert_allclose(
            cal_merged.class_params, cal_full.class_params, atol=1e-8
        )
        np.testing.assert_allclose(
            cal_merged.global_params, cal_full.global_params, atol=1e-8
        )
        # Empty shards are dropped, not counted.
        empty = (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
        merged2 = merge_calibration_data([empty] + triples + [empty])
        self.assertEqual(merged2[0].size, merged[0].size)


class TestDomainShift(unittest.TestCase):
    def test_shift_report_identifies_class_and_magnitude(self):
        rng = np.random.default_rng(53)
        ref_s0, new_s0 = rng.beta(2, 2, 2000), rng.beta(2, 2, 2000)
        ref_s1 = rng.beta(2, 2, 2000)
        new_s1 = rng.beta(6, 2, 2000)  # class 1 drifts upward
        ref_scores = np.concatenate([ref_s0, ref_s1])
        ref_classes = np.array([0] * 2000 + [1] * 2000)
        new_scores = np.concatenate([new_s0, new_s1])
        new_classes = np.array([0] * 2000 + [1] * 2000)
        report = detect_domain_shift(
            ref_scores, ref_classes, new_scores, new_classes, num_classes=2,
            ks_threshold=0.15,
        )
        self.assertTrue(report.shifted)
        self.assertEqual(report.shifted_classes, [1])
        entry = report.entries[1]
        self.assertGreater(entry.ks, 0.15)
        self.assertGreater(entry.mean_new, entry.mean_reference)
        text = report.format()
        self.assertIn("YES", text)
        print("\n[shift] per-class domain shift:")
        print(text)


class TestInstancesIntegration(unittest.TestCase):
    def _make_image(self, rng, n_pred=8, n_gt=5, num_classes=2):
        image_size = (100, 100)
        gt = Instances(image_size)
        gt_boxes = torch.tensor(
            [[10 * i, 10 * i, 10 * i + 8, 10 * i + 8] for i in range(n_gt)],
            dtype=torch.float32,
        )
        gt.set("gt_boxes", Boxes(gt_boxes))
        gt.set("gt_classes", torch.randint(0, num_classes, (n_gt,)))

        pred = Instances(image_size)
        jitter = rng.normal(0, 1.0, (n_pred, 4)).astype(np.float32)
        base = gt_boxes[rng.integers(0, n_gt, n_pred)]
        pred_boxes = base + torch.from_numpy(jitter)
        pred.set("pred_boxes", Boxes(pred_boxes))
        pred.set("scores", torch.rand(n_pred))
        pred.set("pred_classes", gt.get("gt_classes")[rng.integers(0, n_gt, n_pred)])
        return pred, gt

    def test_fit_and_calibrate_instances(self):
        rng = np.random.default_rng(59)
        preds, gts = zip(*[self._make_image(rng) for _ in range(40)])
        calibrator, report = fit_calibrator(list(preds), list(gts), num_classes=2)
        self.assertIsNotNone(calibrator)
        total = sum(c.n_positives + c.n_negatives for c in report.classes)
        self.assertEqual(total, 40 * 8)

        inst = preds[0]
        self.assertFalse(inst.has(CALIBRATED_SCORE_FIELD))
        fields_before = set(inst.get_fields().keys())
        calibrate_instances(inst, calibrator)
        self.assertTrue(inst.has(CALIBRATED_SCORE_FIELD))
        # Existing fields untouched; the new field is additive only.
        self.assertEqual(set(inst.get_fields().keys()), fields_before | {CALIBRATED_SCORE_FIELD})
        probs = inst.get(CALIBRATED_SCORE_FIELD)
        self.assertEqual(probs.shape, inst.get("scores").shape)
        self.assertTrue(bool(((probs >= 0) & (probs <= 1)).all()))

    def test_uncalibrated_path_is_noop(self):
        rng = np.random.default_rng(61)
        pred, _ = self._make_image(rng)
        calibrate_instances(pred, None)
        self.assertFalse(pred.has(CALIBRATED_SCORE_FIELD))

    def test_no_ground_truth_anywhere(self):
        rng = np.random.default_rng(67)
        preds, _ = zip(*[self._make_image(rng, n_gt=1) for _ in range(3)])
        gts = []
        for _ in preds:
            gt = Instances((100, 100))
            gt.set("gt_boxes", Boxes(torch.zeros(0, 4)))
            gt.set("gt_classes", torch.zeros(0, dtype=torch.int64))
            gts.append(gt)
        with self.assertLogs("detectron2.calibration.api", level="INFO"):
            calibrator, report = fit_calibrator(list(preds), gts)
        self.assertIsNone(calibrator)
        self.assertIsNone(report)

    def test_gather_single_image(self):
        rng = np.random.default_rng(71)
        pred, gt = self._make_image(rng, n_pred=1, n_gt=1)
        scores, labels, classes = gather_calibration_data([pred], [gt])
        self.assertEqual(scores.shape, (1,))
        self.assertEqual(labels.shape, (1,))


if __name__ == "__main__":
    unittest.main()
