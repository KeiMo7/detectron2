# Copyright (c) Facebook, Inc. and its affiliates.
"""Test-only dependency shims.

The CI/dev image this runs in lacks fvcore, pycocotools and torchvision, which
the full detectron2 package imports at load time. The tiling package itself only
needs torch/numpy plus the leaf structure classes; these stubs satisfy the
import chain without changing any production behaviour.
"""

import sys
import types


def install_shims() -> None:
    # ---- fvcore ----
    if "fvcore" not in sys.modules:
        fv = types.ModuleType("fvcore")
        fv_nn = types.ModuleType("fvcore.nn")
        fv_dist = types.ModuleType("fvcore.nn.distributed")
        fv_dist.differentiable_all_reduce = lambda x: x
        fv_nn.distributed = fv_dist
        fv.nn = fv_nn
        fv_common = types.ModuleType("fvcore.common")
        fv_config = types.ModuleType("fvcore.common.config")

        class _CfgNode(dict):
            def clone(self):
                return _CfgNode(self)

            def defrost(self):
                return self

            def merge_from_other_cfg(self, *_):
                return self

            def merge_from_file(self, *_):
                return self

        fv_config.CfgNode = _CfgNode
        fv_common.config = fv_config
        fv.common = fv_common
        sys.modules.update(
            {
                "fvcore": fv,
                "fvcore.nn": fv_nn,
                "fvcore.nn.distributed": fv_dist,
                "fvcore.common": fv_common,
                "fvcore.common.config": fv_config,
            }
        )

    # ---- pycocotools (only mask encode/decode entry points are imported) ----
    if "pycocotools" not in sys.modules:
        coco = types.ModuleType("pycocotools")
        mask_util = types.ModuleType("pycocotools.mask")

        def _missing(*_a, **_kw):
            raise RuntimeError("pycocotools stub: polygon ops unavailable in tests")

        for name in ("encode", "decode", "merge", "area", "toBbox", "frPyObjects"):
            setattr(mask_util, name, _missing)
        coco.mask = mask_util
        sys.modules["pycocotools"] = coco
        sys.modules["pycocotools.mask"] = mask_util

    # ---- torchvision (ROIAlign import chain) ----
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv_ops = types.ModuleType("torchvision.ops")

        class _ROIAlign:  # never exercised by tiling tests
            def __init__(self, *a, **k):
                pass

        def _roi_align(*a, **k):  # pragma: no cover - unused in tiling tests
            raise RuntimeError("torchvision stub: roi_align unavailable in tests")

        tv_ops.ROIAlign = _ROIAlign
        tv_ops.roi_align = _roi_align
        tv.ops = tv_ops
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.ops"] = tv_ops

    # ---- detectron2 layer leaf modules used by structures.masks ----
    import importlib
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Preload leaf structure modules directly so the package __init__ (which
    # pulls the whole layers stack) is bypassed.
    def _load(modname, relpath):
        if modname in sys.modules:
            return sys.modules[modname]
        spec = importlib.util.spec_from_file_location(
            modname, os.path.join(repo_root, relpath)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg = types.ModuleType("detectron2")
    pkg.__path__ = [os.path.join(repo_root, "detectron2")]
    sys.modules.setdefault("detectron2", pkg)

    struct_pkg = types.ModuleType("detectron2.structures")
    struct_pkg.__path__ = [os.path.join(repo_root, "detectron2", "structures")]
    sys.modules["detectron2.structures"] = struct_pkg

    _load("detectron2.structures.boxes", "detectron2/structures/boxes.py")
    _load("detectron2.structures.instances", "detectron2/structures/instances.py")
    _load("detectron2.structures.keypoints", "detectron2/structures/keypoints.py")
    # masks.py imports detectron2.layers.roi_align and detectron2.utils.memory
    layers_pkg = types.ModuleType("detectron2.layers")
    layers_pkg.__path__ = [os.path.join(repo_root, "detectron2", "layers")]
    sys.modules["detectron2.layers"] = layers_pkg
    utils_pkg = types.ModuleType("detectron2.utils")
    utils_pkg.__path__ = [os.path.join(repo_root, "detectron2", "utils")]
    sys.modules["detectron2.utils"] = utils_pkg
    _load("detectron2.layers.roi_align", "detectron2/layers/roi_align.py")
    _load("detectron2.utils.memory", "detectron2/utils/memory.py")
    masks_mod = _load("detectron2.structures.masks", "detectron2/structures/masks.py")

    # Expose the leaf classes through the structures package namespace, which
    # is what production code imports (`from detectron2.structures import ...`).
    from detectron2.structures.boxes import Boxes, BoxMode
    from detectron2.structures.instances import Instances
    from detectron2.structures.keypoints import Keypoints

    struct_pkg.Boxes = Boxes
    struct_pkg.BoxMode = BoxMode
    struct_pkg.Instances = Instances
    struct_pkg.Keypoints = Keypoints
    struct_pkg.BitMasks = masks_mod.BitMasks
