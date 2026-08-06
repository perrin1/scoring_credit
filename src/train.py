"""Entrainement, comparaison des modeles et serialisation (parties D et E3/E4).

Usage : python src/train.py
Produit : models/pipeline_scoring.joblib  (pipeline calibre + seuil + metadonnees)
          reports/comparaison_modeles.csv
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import preparer

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
COUT_FN, COUT_FP = 5, 1


EXCLURE = ["defaut_paiement", "id_client", "date_octroi", "annee_octroi",
           "periode_octroi", "agent_credit"]


def charger():
    df = preparer(pd.read_csv(RACINE / "data" / "credits_microfinance.csv"))
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLURE]
    cat = [c for c in df.select_dtypes(include="object").columns if c not in EXCLURE]
    return df, num, cat


def construire_prepro(cols_num, cols_cat):
    """Imputation et encodage appris DANS le pipeline (consigne D2)."""
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols_num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cols_cat),
    ])


def decouper(df, X, y, part=0.8):
    """Decoupe temporelle sur une DATE seuil """
    seuil = df["date_octroi"].quantile(part)
    train = df["date_octroi"] < seuil
    return X[train], y[train], X[~train], y[~train], seuil


def seuil_optimal(y_vrai, proba, cout_fn=COUT_FN, cout_fp=COUT_FP):
    """Seuil qui minimise cout_fn * FN + cout_fp * FP (consigne E3)."""
    seuils = np.linspace(0.01, 0.99, 197)
    couts = []
    for s in seuils:
        _, fp, fn, _ = confusion_matrix(y_vrai, (proba >= s).astype(int), labels=[0, 1]).ravel()
        couts.append(cout_fn * fn + cout_fp * fp)
    couts = np.array(couts)
    return seuils[couts.argmin()], couts, seuils


def experience_fuite(cv):
    """Consigne section 3 : le meme modele avec et sans les colonnes post-decision.  """
    brut = pd.read_csv(RACINE / "data" / "credits_microfinance.csv")
    lignes = []
    for garder_fuite in (True, False):
        df = preparer(brut, garder_fuite=garder_fuite)
        num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLURE]
        cat = [c for c in df.select_dtypes(include="object").columns if c not in EXCLURE]
        X, y = df[num + cat], df["defaut_paiement"]
        X_tr, y_tr, X_te, y_te, _ = decouper(df, X, y)
        pipe = Pipeline([("pre", construire_prepro(num, cat)),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=RANDOM_STATE))])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        lignes.append({"configuration": "AVEC les colonnes de fuite" if garder_fuite
                       else "SANS (modele retenu)",
                       "cv_roc_auc": round(scores.mean(), 4),
                       "test_roc_auc": round(roc_auc_score(y_te, proba), 4),
                       "test_pr_auc": round(average_precision_score(y_te, proba), 4)})
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(RACINE / "reports" / "experience_fuite.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def comparer_desequilibre(cols_num, cols_cat, X_train, y_train, X_test, y_test, cv):
    """Consigne D5 : quatre traitements du desequilibre, a modele egal.

    ImbPipeline (imbalanced-learn) et non Pipeline (sklearn) : le
    reechantillonnage ne doit s'appliquer qu'aux blocs d'entrainement, jamais
    aux blocs de validation, sinon on evalue sur des individus synthetiques.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    log = lambda **kw: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **kw)
    approches = {
        "1. aucun traitement": Pipeline([("pre", construire_prepro(cols_num, cols_cat)),
                                         ("clf", log())]),
        "2. class_weight='balanced'": Pipeline([("pre", construire_prepro(cols_num, cols_cat)),
                                                ("clf", log(class_weight="balanced"))]),
        "3. sous-echantillonnage": ImbPipeline([("pre", construire_prepro(cols_num, cols_cat)),
                                                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                                                ("clf", log())]),
        "4. SMOTE": ImbPipeline([("pre", construire_prepro(cols_num, cols_cat)),
                                 ("sampler", SMOTE(random_state=RANDOM_STATE)),
                                 ("clf", log())]),
    }
    lignes = []
    for nom, pipe in approches.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        lignes.append({"approche": nom, "cv_roc_auc": round(scores.mean(), 4),
                       "cv_ecart_type": round(scores.std(), 4),
                       "test_roc_auc": round(roc_auc_score(y_test, proba), 4)})
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(RACINE / "reports" / "comparaison_desequilibre.csv", index=False)
    print(tableau.to_string(index=False))
    return tableau


def regler_hyperparametres(cols_num, cols_cat, X_train, y_train, X_test, y_test, cv):
    """Consigne D6 : RandomizedSearchCV, grille documentee.

    """
    from sklearn.model_selection import RandomizedSearchCV

    grille = {
        "clf__C": np.logspace(-3, 2, 20),          # force de regularisation
        "clf__l1_ratio": [0, 0.5, 1],              # ridge / elasticnet / lasso
        "clf__class_weight": ["balanced", None],
    }
    recherche = RandomizedSearchCV(
        Pipeline([("pre", construire_prepro(cols_num, cols_cat)),
                  ("clf", LogisticRegression(solver="saga", max_iter=3000,
                                             random_state=RANDOM_STATE))]),
        grille, n_iter=30, cv=cv, scoring="roc_auc",
        random_state=RANDOM_STATE, refit=True)
    recherche.fit(X_train, y_train)
    proba = recherche.predict_proba(X_test)[:, 1]
    print(f"  grille : {grille}")
    print(f"  meilleurs parametres : {recherche.best_params_}")
    print(f"  CV {recherche.best_score_:.4f} | test {roc_auc_score(y_test, proba):.4f}")
    pd.DataFrame(recherche.cv_results_).to_csv(
        RACINE / "reports" / "recherche_hyperparametres.csv", index=False)
    return recherche


def main():
    df, cols_num, cols_cat = charger()
    X, y = df[cols_num + cols_cat], df["defaut_paiement"]
    X_train, y_train, X_test, y_test, date_seuil = decouper(df, X, y)
    print(f"train {len(X_train)} / test {len(X_test)}  (frontiere {date_seuil.date()})")

    candidats = {
        "Regression logistique": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Foret aleatoire": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    resultats, modeles = [], {}
    for nom, estimateur in candidats.items():
        pipe = Pipeline([("pre", construire_prepro(cols_num, cols_cat)), ("clf", estimateur)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        resultats.append({
            "modele": nom,
            "cv_roc_auc": round(scores.mean(), 4),
            "cv_ecart_type": round(scores.std(), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        })
        modeles[nom] = pipe
        print(f"  {nom:<24} CV {scores.mean():.4f} +/- {scores.std():.4f}"
              f"   test {roc_auc_score(y_test, proba):.4f}")

    tableau = pd.DataFrame(resultats).sort_values("cv_roc_auc", ascending=False)
    (RACINE / "reports").mkdir(exist_ok=True)
    tableau.to_csv(RACINE / "reports" / "comparaison_modeles.csv", index=False)

    meilleur_nom = tableau.iloc[0]["modele"]
    print(f"\nMeilleur modele : {meilleur_nom}")

    print("\n--- Consigne section 3 : experience de fuite de donnees ---")
    experience_fuite(cv)

    print("\n--- D5 : traitement du desequilibre ---")
    comparer_desequilibre(cols_num, cols_cat, X_train, y_train, X_test, y_test, cv)

    print("\n--- D6 : reglage des hyperparametres ---")
    regler_hyperparametres(cols_num, cols_cat, X_train, y_train, X_test, y_test, cv)
    print()

    # E4 — calibration : class_weight='balanced' gonfle les probabilites.
    # Le modele livre a l'app doit annoncer des probabilites fiables.
    final = CalibratedClassifierCV(
        Pipeline([("pre", construire_prepro(cols_num, cols_cat)),
                  ("clf", candidats[meilleur_nom])]),
        method="isotonic", cv=5).fit(X_train, y_train)
    proba_cal = final.predict_proba(X_test)[:, 1]

    # E3 — le seuil se calcule APRES calibration, sur l'echelle finale.
    seuil, _, _ = seuil_optimal(y_test, proba_cal)
    _, fp, fn, _ = confusion_matrix(y_test, (proba_cal >= seuil).astype(int), labels=[0, 1]).ravel()
    print(f"Seuil optimal (cout {COUT_FN}:{COUT_FP}) : {seuil:.3f}"
          f"   cout {COUT_FN*fn + COUT_FP*fp}   AUC {roc_auc_score(y_test, proba_cal):.4f}")

    (RACINE / "models").mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": final,
        "seuil": float(seuil),
        "cols_num": cols_num,
        "cols_cat": cols_cat,
        "modele": meilleur_nom,
        "date_seuil": str(date_seuil.date()),
        "cout_fn": COUT_FN,
        "cout_fp": COUT_FP,
        "test_roc_auc": float(roc_auc_score(y_test, proba_cal)),
        "test_pr_auc": float(average_precision_score(y_test, proba_cal)),
    }, RACINE / "models" / "pipeline_scoring.joblib")
    print(f"-> models/pipeline_scoring.joblib")


if __name__ == "__main__":
    main()
