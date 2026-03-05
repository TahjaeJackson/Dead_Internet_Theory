import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

# CALIBRATION METRICS


def calibration_metrics(y_true, probs, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins)

    # Observed-to-Expected ratio
    o_e = probs.sum() / y_true.sum() if y_true.sum() > 0 else np.nan

    # Calibration slope
    slope = np.polyfit(mean_pred, frac_pos, 1)[0] if len(mean_pred) > 1 else np.nan

    # Integrated Calibration Index
    ici = np.mean(np.abs(frac_pos - mean_pred))

    return o_e, slope, ici

# THRESHOLD METRICS


def threshold_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return sensitivity, specificity, ppv, npv


# PLOTTING FUNCTIONS

def plot_roc(y_true, probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr(y_true, probs, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, probs, save_path=None):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction")
    plt.title("Calibration Curve")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_decision_curve(y_true, probs, save_path=None):
    thresholds = np.linspace(0.01, 0.5, 100)
    N = len(y_true)

    net_benefits = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        net_benefit = (tp / N) - (fp / N) * (t / (1 - t))
        net_benefits.append(net_benefit)

    plt.figure()
    plt.plot(thresholds, net_benefits, label="Model")
    plt.axhline(0, linestyle="--", label="Treat None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_tradeoff(y_true, probs, save_path=None):
    thresholds = np.linspace(0.01, 0.5, 100)

    sensitivities = []
    specificities = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    plt.figure()
    plt.plot(thresholds, sensitivities, label="Sensitivity")
    plt.plot(thresholds, specificities, label="Specificity")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Tradeoff")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# MAIN EVALUATION FUNCTION


def evaluate_model(model, X, y, threshold=0.5, plot_prefix=None):

    probs = model.predict_proba(X)[:, 1]

    auroc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)
    ipa = 1 - (brier / np.mean((y - y.mean()) ** 2))
    logloss = log_loss(y, probs)

    o_e, slope, ici = calibration_metrics(y, probs)
    sens, spec, ppv, npv = threshold_metrics(y, probs, threshold)

    if plot_prefix:
        plot_roc(y, probs, f"{plot_prefix}_roc.png")
        plot_pr(y, probs, f"{plot_prefix}_pr.png")
        plot_calibration(y, probs, f"{plot_prefix}_calibration.png")
        plot_decision_curve(y, probs, f"{plot_prefix}_decision_curve.png")
        plot_threshold_tradeoff(y, probs, f"{plot_prefix}_threshold_tradeoff.png")

    return {
        "AUROC": auroc,
        "PR_AUC": pr_auc,
        "O_E": o_e,
        "Calibration_Slope": slope,
        "ICI": ici,
        "Brier": brier,
        "IPA": ipa,
        "LogLoss": logloss,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "Prevalence": y.mean(),
        "y_prob": probs,
        "y_true": y.values
    }
