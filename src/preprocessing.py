
import numpy as np
import pandas as pd

# --- Colonnes connues seulement APRES le defaut : fuite de donnees ---
COLS_FUITE = ["nb_relances_recouvrement", "statut_dossier"]

# --- Sentinelles detectees en A3 ---
SENTINELLES = {"age": [-1], "revenu_mensuel_fcfa": [-999], "distance_agence_km": [9999]}

COLS_CAT = ["sexe", "situation_matrimoniale", "niveau_education", "zone_habitation",
            "secteur_activite", "objet_credit", "type_garantie"]

SYNONYMES = {"sexe": {"Feminin": "F", "Masculin": "M"},
             "situation_matrimoniale": {"Marie(e)": "Marie"}}

SEUIL_ERREUR_UNITE = 2_000_000


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les corrections des parties A2 a A6. Ne modifie pas l'entree."""
    df = df.copy()

    # A4  harmonisation des categorielles (avant A2 : le parsing des dates
    # revele 3 doublons supplementaires masques par le double format)
    for col in COLS_CAT:
        if col in df:
            df[col] = (df[col].str.strip()
                              .str.replace(r"\s+", " ", regex=True)
                              .str.capitalize()
                              .replace(SYNONYMES.get(col, {})))
    if "agent_credit" in df:
        df["agent_credit"] = df["agent_credit"].str.strip().str.upper()

    if "date_octroi" in df and df["date_octroi"].dtype == "object":
        df["date_octroi"] = pd.to_datetime(df["date_octroi"], format="mixed", dayfirst=True)

    df = df.drop_duplicates().reset_index(drop=True)

    for col, valeurs in SENTINELLES.items():
        if col in df:
            df[col] = df[col].replace(valeurs, np.nan)

    if "revenu_mensuel_fcfa" in df:
        masque = df["revenu_mensuel_fcfa"] > SEUIL_ERREUR_UNITE
        df.loc[masque, "revenu_mensuel_fcfa"] /= 1000

    if {"anciennete_activite_mois", "age"} <= set(df.columns):
        plafond = (df["age"] - 15) * 12
        df["anciennete_activite_mois"] = df["anciennete_activite_mois"].where(
            df["anciennete_activite_mois"] <= plafond, plafond)

    return df.drop(columns=[c for c in COLS_FUITE if c in df])


def ajouter_variables_derivees(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    rev, ch = df["revenu_mensuel_fcfa"], df["charges_mensuelles_fcfa"]
    montant, duree = df["montant_credit_fcfa"], df["duree_credit_mois"]


    df["capacite_remboursement"] = (rev - ch).clip(lower=1)
    df["mensualite_estimee"] = montant * (1 + df["taux_interet_annuel"]/100 * duree/12) / duree
    df["ratio_endettement"] = df["mensualite_estimee"] / df["capacite_remboursement"]
    df["taux_retard_historique"] = df["nb_retards_anterieurs"] / df["nb_credits_anterieurs"].clip(lower=1)
    df["credit_sur_revenu"] = montant / rev
    df["revenu_par_personne"] = rev / (df["nb_personnes_charge"] + 1)


    df["ratio_charges"] = ch / rev
    df["epargne_sur_credit"] = df["montant_epargne_fcfa"] / montant
    df["primo_emprunteur"] = (df["nb_credits_anterieurs"] == 0).astype(int)


    df["log_ratio_endettement"] = np.log1p(df["ratio_endettement"].clip(lower=0))

    df["agri_credit_court"] = ((df["secteur_activite"] == "Agriculture")
                               & (duree <= 6)).astype(int)

    df["couverture_epargne"] = df["montant_epargne_fcfa"] / df["mensualite_estimee"]


    df["score_mm_manquant"] = df["score_mobile_money"].isna().astype(int)
    df["revenu_manquant"] = df["revenu_mensuel_fcfa"].isna().astype(int)


    if "date_octroi" in df:
        df["mois_octroi"] = df["date_octroi"].dt.month
        df["trimestre_octroi"] = df["date_octroi"].dt.quarter

    return df.replace([np.inf, -np.inf], np.nan)


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Chaine complete : nettoyage puis variables derivees."""
    return ajouter_variables_derivees(nettoyer(df))
