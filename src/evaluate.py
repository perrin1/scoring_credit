"""Evaluation complete du modele serialise partie E.

Usage : python src/evaluate.py   (apres python src/train.py)
Produit : reports/e1_matrice_confusion.png, e3_cout_seuil.png,
          e4_calibration.png, e5_importance.png, e6_equite.png,
          reports/metriques.csv
"""
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from preprocessing import preparer
from train import EXCLURE, decouper, seuil_optimal

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent
REPORTS = RACINE / "reports"
BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#52514e"


def habiller(ax, titre="", x="", yl=""):
    ax.set_title(titre, fontsize=11)
    ax.set_xlabel(x); ax.set_ylabel(yl)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)


def main():
    REPORTS.mkdir(exist_ok=True)
    bundle = joblib.load(RACINE / "models" / "pipeline_scoring.joblib")
    modele, seuil = bundle["pipeline"], bundle["seuil"]
    cout_fn, cout_fp = bundle["cout_fn"], bundle["cout_fp"]

    df = preparer(pd.read_csv(RACINE / "data" / "credits_microfinance.csv"))
    X = df[bundle["cols_num"] + bundle["cols_cat"]]
    y = df["defaut_paiement"]
    X_train, y_train, X_test, y_test, _ = decouper(df, X, y)
    proba = modele.predict_proba(X_test)[:, 1]
    test = df[df["date_octroi"] >= df["date_octroi"].quantile(0.8)].copy()

    # ---------------- E1 : metriques et matrice de confusion ----------------
    lignes = []
    for nom, s in [("seuil 0,50 (defaut)", 0.5), (f"seuil {seuil:.2f} (cout metier)", seuil)]:
        pred = (proba >= s).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        lignes.append({
            "configuration": nom, "seuil": round(s, 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "rappel": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "taux_refus": round(pred.mean(), 3),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "cout": cout_fn * fn + cout_fp * fp,
        })
    metriques = pd.DataFrame(lignes)
    metriques.to_csv(REPORTS / "metriques.csv", index=False)
    print(metriques.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    pred = (proba >= seuil).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]}", ha="center", va="center",
                         fontsize=15, color="white" if cm[i, j] > cm.max()/2 else "black")
    axes[0].set_xticks([0, 1], ["accepte", "refuse"])
    axes[0].set_yticks([0, 1], ["sain", "defaut"])
    habiller(axes[0], f"Matrice de confusion (seuil {seuil:.2f})", "prediction", "realite")
    axes[0].grid(False)

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, color=BLEU, lw=2,
                 label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color=GRIS, lw=1, label="hasard = 0,500")
    habiller(axes[1], "Courbe ROC", "taux de faux positifs", "rappel"); axes[1].legend(frameon=False)

    prec, rapp, _ = precision_recall_curve(y_test, proba)
    axes[2].plot(rapp, prec, color=ORANGE, lw=2,
                 label=f"PR AUC = {average_precision_score(y_test, proba):.3f}")
    axes[2].axhline(y_test.mean(), ls="--", color=GRIS, lw=1,
                    label=f"hasard = {y_test.mean():.3f}")
    habiller(axes[2], "Courbe precision-rappel", "rappel", "precision"); axes[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORTS / "e1_metriques.png", dpi=150, bbox_inches="tight")

    # ---------------- E3 : cout en fonction du seuil ----------------
    s_opt, couts, seuils = seuil_optimal(y_test, proba, cout_fn, cout_fp)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(seuils, couts, color=BLEU, lw=2)
    ax.axvline(0.5, ls="--", color=GRIS, lw=1.5,
               label=f"seuil 0,50 -> cout {couts[np.abs(seuils-0.5).argmin()]}")
    ax.axvline(s_opt, ls="--", color=ORANGE, lw=1.5,
               label=f"optimal {s_opt:.2f} -> cout {couts.min()}")
    habiller(ax, f"Cout metier = {cout_fn}xFN + {cout_fp}xFP", "seuil de decision", "cout total")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORTS / "e3_cout_seuil.png", dpi=150, bbox_inches="tight")
    print(f"\nE3 : seuil optimal {s_opt:.3f} (cout {couts.min()}) "
          f"vs 0,50 (cout {couts[np.abs(seuils-0.5).argmin()]})")

    # ---------------- E4 : calibration ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    obs, pred_moy = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    ax.plot(pred_moy, obs, "o-", color=BLEU, label=f"modele (Brier {brier_score_loss(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1, label="calibration parfaite")
    habiller(ax, "Courbe de calibration", "probabilite predite", "frequence observee")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORTS / "e4_calibration.png", dpi=150, bbox_inches="tight")
    print(f"E4 : Brier {brier_score_loss(y_test, proba):.4f}")

    # ---------------- E5 : importance par permutation ----------------
    imp = permutation_importance(modele, X_test, y_test, n_repeats=10,
                                 random_state=42, scoring="roc_auc")
    top = (pd.DataFrame({"variable": X.columns, "importance": imp.importances_mean,
                         "sd": imp.importances_std})
           .sort_values("importance", ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["variable"][::-1], top["importance"][::-1],
            xerr=top["sd"][::-1], color=BLEU, height=.7)
    habiller(ax, "Importance par permutation (chute de ROC AUC sur le test)", "chute d'AUC", "")
    fig.tight_layout(); fig.savefig(REPORTS / "e5_importance.png", dpi=150, bbox_inches="tight")
    print("\nE5 :"); print(top.head(8).to_string(index=False))

    # ---------------- E6 : equite ----------------
    test["proba"] = proba
    test["refuse"] = (proba >= seuil).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col in zip(axes, ["sexe", "zone_habitation"]):
        g = test.groupby(col).apply(lambda d: pd.Series({
            "taux de defaut reel": d["defaut_paiement"].mean(),
            "taux de refus": d["refuse"].mean(),
            "rappel": recall_score(d["defaut_paiement"], d["refuse"], zero_division=0),
        }), include_groups=False)
        print(f"\nE6 - {col} :"); print(g.round(3).to_string())
        x = np.arange(len(g))
        for i, (c, coul) in enumerate(zip(g.columns, [GRIS, ORANGE, BLEU])):
            ax.bar(x + (i - 1) * .27, g[c], width=.25, label=c, color=coul)
        ax.set_xticks(x, g.index)
        habiller(ax, f"Equite selon {col}", "", "proportion")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "e6_equite.png", dpi=150, bbox_inches="tight")
    print(f"\n-> figures ecrites dans {REPORTS}")


if __name__ == "__main__":
    main()
