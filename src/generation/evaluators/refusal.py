import string
from typing import Any


REFUSAL_KEYWORDS = (
    "insufficient context",
    "context is insufficient",
    "insufficient information",
    "onvoldoende context",
    "de context is onvoldoende",
    "is de context onvoldoende",
    "de context bieden geen informatie",
    "de context bevat onvoldoende",
    "uit de context kan niet worden afgeleid",
    "de context is onvoldoende om de vraag te beantwoorden",
    "de context ontoereikend",
    "in de verstrekte context ontbreekt",
    "de context bevat geen specifieke bepaling",
    "contexte insuffisant",
    "contexte est insuffisant",
    "le contexte est insuffisant",
    "le contexte fourni est insuffisant",
    "le contexte juridique fourni est insuffisant",
    "le contexte est insuffisant pour répondre à la question",
    "le contexte est insuffisant pour repondre a la question",
)


def _looks_like_refusal(text: object) -> bool:
    if text is None:
        return False
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        return False
    stripped = normalized.translate(str.maketrans("", "", string.punctuation))
    if stripped in {
        "insufficient context",
        "context is insufficient",
        "onvoldoende context",
        "de context is onvoldoende",
        "contexte insuffisant",
        "contexte est insuffisant",
    }:
        return True
    return any(keyword in normalized for keyword in REFUSAL_KEYWORDS)


def _contains_refusal_phrase(value: object) -> bool:
    if isinstance(value, str):
        return _looks_like_refusal(value)
    if isinstance(value, list):
        return any(_contains_refusal_phrase(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_refusal_phrase(item) for item in value.values())
    return False


def _is_refusal_output(cleaned: object) -> bool:
    # Loose criterion requested: refusal if any string in the payload contains
    # a refusal phrase, regardless of citation presence.
    return _contains_refusal_phrase(cleaned)


def _evaluate_refusals(
    *,
    gold_by_qid: dict[str, list[str]],
    expected_refusal_by_qid: dict[str, bool],
    retrieved_by_qid: dict[str, list[str]],
    predicted_by_qid: dict[str, list[str]],
    refusals_by_qid: dict[str, bool],
    model_name: str,
    setup_name: str,
) -> dict[str, Any]:
    candidate_ids = sorted(set(gold_by_qid.keys()) & set(predicted_by_qid.keys()))
    refusal_ids = sorted(qid for qid, is_refusal in refusals_by_qid.items() if is_refusal)
    refusal_ids_compared = [qid for qid in candidate_ids if refusals_by_qid.get(qid, False)]
    expected_refusal_ids = [qid for qid in candidate_ids if expected_refusal_by_qid.get(qid, False)]
    expected_refusal_labels = [int(expected_refusal_by_qid.get(qid, False)) for qid in candidate_ids]
    predicted_refusal_labels = [int(refusals_by_qid.get(qid, False)) for qid in candidate_ids]

    refusal_report_dict: dict[str, Any] | None = None
    if candidate_ids:
        from sklearn.metrics import classification_report

        report_kwargs = {
            "labels": [0, 1],
            "target_names": ["not_refusal", "refusal"],
            "zero_division": 0,
        }
        refusal_report_dict = classification_report(
            expected_refusal_labels,
            predicted_refusal_labels,
            output_dict=True,
            **report_kwargs,
        )

    refusal_accuracy = (
        float(refusal_report_dict.get("accuracy")) if isinstance(refusal_report_dict, dict) else None
    )

    report_refusal = (
        refusal_report_dict.get("refusal", {}) if isinstance(refusal_report_dict, dict) else {}
    )
    report_not_refusal = (
        refusal_report_dict.get("not_refusal", {}) if isinstance(refusal_report_dict, dict) else {}
    )
    report_macro = (
        refusal_report_dict.get("macro avg", {}) if isinstance(refusal_report_dict, dict) else {}
    )

    refusal_rate_compared = (
        len(refusal_ids_compared) / len(candidate_ids) if candidate_ids else None
    )
    refusal_table = {
        "Model": model_name,
        "Setup": setup_name,
        "RefRate": refusal_rate_compared,
        "F1-macro": report_macro.get("f1-score"),
        "RefPr": report_refusal.get("precision"),
        "RefRec": report_refusal.get("recall"),
        "RefF1": report_refusal.get("f1-score"),
        "NonRefRPr": report_not_refusal.get("precision"),
        "NonRefRRec": report_not_refusal.get("recall"),
        "NonRefRF1": report_not_refusal.get("f1-score"),
    }

    by_id: dict[str, dict[str, Any]] = {}
    for qid in candidate_ids:
        expected_refusal = expected_refusal_by_qid.get(qid, False)
        predicted_refusal = refusals_by_qid.get(qid, False)
        retrieved_ids = retrieved_by_qid.get(qid, [])
        retrieved_id_set = set(retrieved_ids)
        missing_gold_ids_in_retrieved = [
            article_id for article_id in gold_by_qid.get(qid, []) if article_id not in retrieved_id_set
        ]
        by_id[qid] = {
            "expected_refusal": expected_refusal,
            "predicted_refusal": predicted_refusal,
            "refusal_correct": expected_refusal == predicted_refusal,
            "retrieved_ids": retrieved_ids,
            "missing_gold_ids_in_retrieved": missing_gold_ids_in_retrieved,
        }

    return {
        "metrics": {
            "refusal_accuracy": refusal_accuracy,
            "refusal_classification_report": refusal_report_dict,
            "refusal_table": refusal_table,
            "refusal_rate_predicted": (
                len(refusal_ids) / len(predicted_by_qid) if predicted_by_qid else None
            ),
            "refusal_rate_compared": refusal_rate_compared,
        },
        "counts": {
            "num_refusals": len(refusal_ids),
            "num_refusals_compared": len(refusal_ids_compared),
            "num_expected_refusals_compared": len(expected_refusal_ids),
        },
        "refusal_ids": refusal_ids,
        "expected_refusal_ids_compared": expected_refusal_ids,
        "by_id": by_id,
    }
