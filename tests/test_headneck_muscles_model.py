from types import SimpleNamespace

import numpy as np

from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue.muscle_adipose_tissue import (
    TS_HEADNECK_MUSCLES_MODEL,
    TS_TOTAL_MODEL,
    MuscleAdiposeTissueComputeMetrics,
    MuscleAdiposeTissuePostProcessing,
    MuscleAdiposeTissueSegmentation,
)


def test_headneck_model_is_registered():
    model = Models.model_from_name(TS_HEADNECK_MUSCLES_MODEL)

    assert model is Models.TS_HEADNECK_MUSCLES_V_0_0_1
    assert model.categories == {"muscle": 0, "sat": 1, "vat": 2, "imat": 3}


def test_ts_total_model_is_registered():
    model = Models.model_from_name(TS_TOTAL_MODEL)

    assert model is Models.TS_TOTAL
    assert model.categories == {"muscle": 0, "sat": 1, "vat": 2, "imat": 3}


def test_headneck_labels_map_only_to_muscle_channel():
    segmentation = MuscleAdiposeTissueSegmentation(
        batch_size=1, model_name=TS_HEADNECK_MUSCLES_MODEL
    )
    categories = segmentation.model_type.categories
    pred = np.array(
        [
            [0, 1, 2],
            [3, 0, 23],
        ],
        dtype=np.uint8,
    )

    masks = segmentation._build_totalseg_masks([pred], categories)

    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0][..., categories["muscle"]], pred > 0)
    assert np.count_nonzero(masks[0][..., categories["sat"]]) == 0
    assert np.count_nonzero(masks[0][..., categories["vat"]]) == 0
    assert np.count_nonzero(masks[0][..., categories["imat"]]) == 0


def test_ts_total_labels_map_only_total_muscles_to_muscle_channel():
    segmentation = MuscleAdiposeTissueSegmentation(
        batch_size=1, model_name=TS_TOTAL_MODEL
    )
    categories = segmentation.model_type.categories
    pred = np.array(
        [
            [0, 1, 80, 89],
            [79, 81, 90, 117],
        ],
        dtype=np.uint8,
    )

    masks = segmentation._build_totalseg_masks([pred], categories)

    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0][..., categories["muscle"]], expected)
    assert np.count_nonzero(masks[0][..., categories["sat"]]) == 0
    assert np.count_nonzero(masks[0][..., categories["vat"]]) == 0
    assert np.count_nonzero(masks[0][..., categories["imat"]]) == 0


def test_ts_total_post_processing_keeps_low_hu_muscle_in_muscle_channel():
    model = Models.model_from_name(TS_TOTAL_MODEL)
    pipeline = SimpleNamespace(
        muscle_adipose_tissue_model_type=model,
        muscle_adipose_tissue_model_name=TS_TOTAL_MODEL,
    )

    image = np.full((5, 5), -50.0, dtype=np.float32)
    pred = np.zeros((5, 5, 4), dtype=np.uint8)
    pred[0:4, 0:4, model.categories["muscle"]] = 1

    processed = MuscleAdiposeTissuePostProcessing()(
        pipeline,
        images=[image],
        preds=[pred],
        spacings=[(1.0, 1.0)],
    )
    mask = processed["masks"][0]

    assert np.count_nonzero(mask[..., model.categories["muscle"]]) == 16
    assert np.count_nonzero(mask[..., model.categories["imat"]]) == 0
    assert np.count_nonzero(mask[..., model.categories["sat"]]) == 0
    assert np.count_nonzero(mask[..., model.categories["vat"]]) == 0


def test_headneck_post_processing_moves_low_hu_muscle_to_imat():
    model = Models.model_from_name(TS_HEADNECK_MUSCLES_MODEL)
    pipeline = SimpleNamespace(
        muscle_adipose_tissue_model_type=model,
        muscle_adipose_tissue_model_name=TS_HEADNECK_MUSCLES_MODEL,
    )

    image = np.full((5, 5), -50.0, dtype=np.float32)
    pred = np.zeros((5, 5, 4), dtype=np.uint8)
    pred[0:4, 0:4, model.categories["muscle"]] = 1

    processed = MuscleAdiposeTissuePostProcessing()(
        pipeline,
        images=[image],
        preds=[pred],
        spacings=[(1.0, 1.0)],
    )
    mask = processed["masks"][0]

    assert np.count_nonzero(mask[..., model.categories["muscle"]]) == 0
    assert np.count_nonzero(mask[..., model.categories["imat"]]) == 16
    assert np.count_nonzero(mask[..., model.categories["sat"]]) == 0
    assert np.count_nonzero(mask[..., model.categories["vat"]]) == 0


def test_metrics_return_zero_for_empty_sat_and_vat():
    model = Models.model_from_name(TS_HEADNECK_MUSCLES_MODEL)
    pipeline = SimpleNamespace(
        muscle_adipose_tissue_model_type=model,
        muscle_adipose_tissue_model_name=TS_HEADNECK_MUSCLES_MODEL,
    )
    image = np.full((3, 3), -20.0, dtype=np.float32)
    mask = np.zeros((3, 3, 4), dtype=np.uint8)
    mask[1:, 1:, model.categories["muscle"]] = 1

    metrics = MuscleAdiposeTissueComputeMetrics()(
        pipeline,
        images=[image],
        masks=[mask],
        spacings=[(1.0, 1.0)],
    )["results"][0]

    result = metrics
    assert result["sat"]["Cross-sectional Area (cm^2)"] == 0
    assert result["vat"]["Cross-sectional Area (cm^2)"] == 0
    assert result["sat"]["Hounsfield Unit"] == 0
    assert result["vat"]["Hounsfield Unit"] == 0
