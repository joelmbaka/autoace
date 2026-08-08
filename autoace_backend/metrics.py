from __future__ import annotations

from typing import Any


def emotional_tone_metrics(
    confusion: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute one-vs-rest precision/recall/F1 from an emotional-tone confusion matrix."""
    labels = list(confusion)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []

    for label in labels:
        true_positive = confusion[label].get(label, 0)
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(
            confusion[expected].get(label, 0)
            for expected in labels
            if expected != label
        )
        support = sum(confusion[label].values())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support > 0:
            f1_values.append(f1)

    macro_f1_observed_classes = (
        sum(f1_values) / len(f1_values) if f1_values else 0.0
    )

    all_f1_values = [float(per_class[label]["f1"]) for label in labels]
    macro_f1_all_classes = (
        sum(all_f1_values) / len(all_f1_values) if all_f1_values else 0.0
    )

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label].get(label, 0) for label in labels)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1_observed_classes": macro_f1_observed_classes,
        "macro_f1_all_classes": macro_f1_all_classes,
        "per_class": per_class,
        "note": (
            "The supplied development set does not cover every emotional-tone class; "
            "macro_f1_observed_classes averages only classes with support, while "
            "macro_f1_all_classes includes zero-support classes as zero."
        ),
    }
