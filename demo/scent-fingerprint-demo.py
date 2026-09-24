
"""

This is a standalone, runnable demo script that showcases the core workflow
of the "Scent Fingerprint Engine":

    Raw e-nose sensor signal -> PCA dimensionality reduction -> LDA classification
    -> Structured .scent data file

Note: Since real MEMS electronic-nose hardware isn't available on-site at a
hackathon, this script uses simulated data that follows realistic sensor
response distributions (based on the pulse-heated MOS sensor response
patterns described in [doc4][doc5]), in order to validate the machine
learning classification pipeline. In a real deployment, simply replace
generate_simulated_sensor_data() with an actual sensor acquisition function
-- the rest of the pipeline stays the same.

Dependencies: numpy, scikit-learn (see requirements.txt)
Run: python scent_fingerprint_demo.py
"""

import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Scent classes: 4 representative "memory scents" are simulated.
# Each scent maps to the center of a 12-dimensional Gaussian distribution
# (mirroring the 12-feature input matrix design described in [doc4]).
# ----------------------------------------------------------------------
SCENT_CLASSES = {
    "grandmas_braised_pork_soy_aroma": np.array([1.5, 2.1, 0.6, 3.0, 2.0, 2.3, -0.2, 1.8, 2.7, 1.8, 0.4, 2.2]),
    "old_fashioned_soap":              np.array([0.3, 0.5, 1.8, 0.4, 0.6, 0.3, 2.1, 0.5, 0.4, 2.0, 1.9, 0.3]),
    "petrichor_after_rain":            np.array([2.2, 0.3, 0.5, 0.6, 2.8, 0.4, 0.5, 2.5, 0.3, 0.4, 0.6, 2.6]),
    "lavender_floral":                 np.array([0.6, 2.6, 2.4, 1.9, 0.4, 2.5, 1.7, 0.4, 2.3, 0.5, 2.2, 0.5]),
}
FEATURE_NOISE_STD = 0.35
SAMPLES_PER_CLASS = 40


def generate_simulated_sensor_data():
    """Simulate 12-dimensional MEMS e-nose response vectors for different scents
    under pulse-heating mode."""
    X, y = [], []
    for label, center in SCENT_CLASSES.items():
        samples = center + np.random.normal(0, FEATURE_NOISE_STD, size=(SAMPLES_PER_CLASS, len(center)))
        X.append(samples)
        y.extend([label] * SAMPLES_PER_CLASS)
    return np.vstack(X), np.array(y)


def train_and_evaluate(X, y):
    """PCA dimensionality reduction + LDA classification; returns the trained
    model and test-set evaluation results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    pca.fit(X_train)

    lda = LDA()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels_order = list(SCENT_CLASSES.keys())
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    return pca, lda, acc, cm, labels_order


def print_confusion_matrix(cm, labels):
    print("\n[3/4] Confusion matrix (rows = true class, columns = predicted class):")
    header = "               " + "".join(f"{lbl[:10]:>12}" for lbl in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = "".join(f"{v:>12}" for v in cm[i])
        print(f"{row_label[:14]:>14}{row_str}")


def build_scent_fingerprint_json(lda_model, sample_vector, place_name="Grandma's Kitchen"):
    """Wrap a single classification result into the .scent data structure
    defined in the concept document."""
    probs = lda_model.predict_proba([sample_vector])[0]
    classes = lda_model.classes_
    ranked = sorted(zip(classes, probs), key=lambda t: -t[1])

    fingerprint = {
        "scent_id": f"sc_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "location": {"place_name": place_name},
        "sensor_fingerprint": {
            "sensor_model": "SnO2-MEMS-PulseArray-v2 (simulated)",
            "sensor_type": "MOS_electronic_nose_demo",
            "raw_response_vector": [round(float(v), 3) for v in sample_vector],
            "classification": {
                "top_label": ranked[0][0],
                "confidence": round(float(ranked[0][1]), 4),
                "candidate_labels": [lbl for lbl, _ in ranked[:3]],
            },
        },
        "context": {
            "mood_tag": "warm/nostalgic",
            "free_text": "This is a demo record used to illustrate the .scent file structure.",
        },
        "usage_stats": {"playback_count": 0},
    }
    return fingerprint


def main():
    print("=" * 60)
    print("Project PROUST -- Scent Fingerprint Classification Demo (simulated data)")
    print("=" * 60)

    X, y = generate_simulated_sensor_data()
    print(f"\n[1/4] Generated simulated e-nose dataset: {len(X)} samples, "
          f"{X.shape[1]} features, {len(SCENT_CLASSES)} scent classes")

    t0 = time.perf_counter()
    pca, lda, acc, cm, labels = train_and_evaluate(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[2/4] Model training complete, took {elapsed_ms:.1f} ms")
    print(f"      Explained variance ratio of top-3 PCA components: "
          + ", ".join(f"{v*100:.1f}%" for v in pca.explained_variance_ratio_))
    print(f"      LDA test-set classification accuracy: {acc*100:.2f}%")

    print_confusion_matrix(cm, labels)

    print("\n[4/4] Generating sample scent fingerprint file sample_scent_fingerprint.json:")
    sample_vector = SCENT_CLASSES["grandmas_braised_pork_soy_aroma"] + np.random.normal(0, FEATURE_NOISE_STD, size=12)
    fingerprint = build_scent_fingerprint_json(lda, sample_vector)
    print(json.dumps(fingerprint, ensure_ascii=False, indent=2))

    with open("sample_scent_fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, ensure_ascii=False, indent=2)
    print("\n[OK] Saved to sample_scent_fingerprint.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
