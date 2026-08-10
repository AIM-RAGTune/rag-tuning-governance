from __future__ import annotations

from pathlib import Path


def write_binary_plots(y_true, y_score, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import os
        import tempfile

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "squaresim-mpl"))
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            PrecisionRecallDisplay,
            RocCurveDisplay,
            confusion_matrix,
        )
    except ImportError:
        return {}

    paths = {}
    try:
        RocCurveDisplay.from_predictions(y_true, y_score)
        paths["roc_curve"] = str(output_dir / "roc_curve.png")
        plt.savefig(paths["roc_curve"], dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
    try:
        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        paths["pr_curve"] = str(output_dir / "pr_curve.png")
        plt.savefig(paths["pr_curve"], dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="black")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed probability")
        paths["calibration"] = str(output_dir / "calibration.png")
        plt.savefig(paths["calibration"], dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
    try:
        cm = confusion_matrix(y_true, [1 if s >= 0.5 else 0 for s in y_score], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot()
        paths["confusion_matrix"] = str(output_dir / "confusion_matrix.png")
        plt.savefig(paths["confusion_matrix"], dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
    return paths
