from face_classification.classification_metrics import (
    AccuracyMetric,
    MeanSquaredErrorMetric,
    PrecisionMetric,
    RecallMetric,
)

CLASSIFICATION_METRICS = {
    "accuracy": AccuracyMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "mean_squared_error": MeanSquaredErrorMetric,
}
