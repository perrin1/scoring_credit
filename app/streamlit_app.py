"""Tableau de bord de scoring de credit  partie F.

Lancement :  streamlit run app/streamlit_app.py

Le pipeline serialise et le module de preparation sont partages avec
l'entrainement (src/preprocessing.py) : les variables derivees sont donc
construites de facon strictement identique ici et dans src/train.py.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

from preprocessing import COLS_FUITE, preparer

st.set_page_config(page_title="Scoring credit microfinance", page_icon="", layout="wide")

CHEMIN_MODELE = RACINE / "models" / "pipeline_scoring.joblib"
CHEMIN_DONNEES = RACINE / "data" / "credits_microfinance.csv"
BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#52514e"


DOMAINE = {
    "age": (18, 78), "revenu_mensuel_fcfa": (10_000, 600_000),
    "charges_mensuelles_fcfa": (3_000, 600_000), "montant_credit_fcfa": (25_000, 4_600_000),
    "taux_interet_annuel": (9.0, 37.5), "score_mobile_money": (0.0, 100.0),
    "distance_agence_km": (0.1, 120.0), "anciennete_activite_mois": (1, 405),
}



# Chargement ( gestion du modele absent / des donnees absentes)

@st.cache_resource
def charger_modele():
    if not CHEMIN_MODELE.exists():
        return None
    return joblib.load(CHEMIN_MODELE)


@st.cache_data
def charger_donnees():
    if not CHEMIN_DONNEES.exists():
        return None
    return preparer(pd.read_csv(CHEMIN_DONNEES))


@st.cache_data
def charger_brut():
    """Donnees brutes : sert de gabarit au formulaire (valeurs par defaut)."""
    return pd.read_csv(CHEMIN_DONNEES) if CHEMIN_DONNEES.exists() else None


bundle = charger_modele()
df = charger_donnees()

if bundle is None:
    st.error("**Modele introuvable**  `models/pipeline_scoring.joblib` est absent.")
    st.code("python src/train.py", language="bash")
    st.stop()
if df is None:
    st.error(f"**Donnees introuvables**  `{CHEMIN_DONNEES}` est absent.")
    st.stop()

modele = bundle["pipeline"]
SEUIL_DEFAUT = bundle["seuil"]
COLS_MODELE = bundle["cols_num"] + bundle["cols_cat"]

decoupe = df["date_octroi"].quantile(0.8)
test = df[df["date_octroi"] >= decoupe].copy()
test["proba"] = modele.predict_proba(test[COLS_MODELE])[:, 1]



def cout_metier(y_vrai, proba, seuil, c_fn=None, c_fp=None):
    c_fn = c_fn if c_fn is not None else bundle["cout_fn"]
    c_fp = c_fp if c_fp is not None else bundle["cout_fp"]
    tn, fp, fn, tp = confusion_matrix(y_vrai, (proba >= seuil).astype(int),
                                      labels=[0, 1]).ravel()
    return c_fn * fn + c_fp * fp, (tn, fp, fn, tp)


st.title(" Scoring de credit  microfinance")
st.caption(f"Modele : **{bundle['modele']}** calibre · decoupe temporelle au "
           f"{bundle['date_seuil']} · ROC AUC test {bundle['test_roc_auc']:.3f} · "
           f"PR AUC {bundle['test_pr_auc']:.3f}")

onglets = st.tabs([" Exploration", " Performance", " Simulateur",
                   " Scoring par lot", "️ Equite"])


# F1  Exploration

with onglets[0]:
    st.subheader("Taux de defaut par segment")
    base = df["defaut_paiement"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dossiers", f"{len(df):,}".replace(",", " "))
    c2.metric("Taux de defaut global", f"{base*100:.1f} %")
    c3.metric("Periode", f"{df['date_octroi'].min():%Y-%m} - {df['date_octroi'].max():%Y-%m}")

    segment = st.selectbox("Segment", ["duree_credit_mois", "secteur_activite",
                                       "type_garantie", "zone_habitation",
                                       "objet_credit", "niveau_education"])
    tableau = (df.groupby(segment)
                 .agg(n=("defaut_paiement", "size"), taux=("defaut_paiement", "mean")))
    tableau["lift"] = (tableau["taux"] / base).round(2)
    tableau["taux"] = (tableau["taux"] * 100).round(1)
    tableau = tableau.sort_values("taux", ascending=False)

    g, d = st.columns([2, 1])
    g.bar_chart(tableau["taux"], color=ORANGE, height=320,
                y_label="taux de defaut (%)")
    d.dataframe(tableau, width='stretch')
    st.caption(f"`lift` = taux du segment / taux global ({base*100:.1f} %). "
               "Au-dessus de 1, le segment est plus risque que la moyenne.")

    st.subheader("Distribution comparee  sains contre defauts")
    variable = st.selectbox("Variable", sorted(bundle["cols_num"]), index=0, key="dist")
    comparaison = pd.DataFrame({
        "sain": df.loc[df.defaut_paiement == 0, variable].describe(),
        "defaut": df.loc[df.defaut_paiement == 1, variable].describe(),
    }).round(2)
    g, d = st.columns([2, 1])
    bornes = df[variable].quantile([0.01, 0.99])
    extrait = df[df[variable].between(*bornes)].copy()

    tranches = pd.qcut(extrait[variable], 20, duplicates="drop").astype(str)
    profil = extrait.groupby(tranches, observed=True)["defaut_paiement"].mean() * 100
    g.bar_chart(profil, color=BLEU, height=300, y_label="taux de defaut (%)",
                x_label=f"{variable} (tranches de 5 %)")
    d.dataframe(comparaison, width='stretch')


# F2  Performance

with onglets[1]:
    st.subheader("Performance sur le jeu de test (20 % les plus recents)")
    y_test = test["defaut_paiement"]

    st.markdown("**Curseur de seuil**  deplace-le pour voir l'impact sur le cout metier.")
    c1, c2 = st.columns([3, 1])
    seuil = c1.slider("Seuil de decision", 0.01, 0.99, float(SEUIL_DEFAUT), 0.01)
    c2.metric("Seuil optimal", f"{SEUIL_DEFAUT:.2f}")

    cout, (tn, fp, fn, tp) = cout_metier(y_test, test["proba"], seuil)
    cout_ref, _ = cout_metier(y_test, test["proba"], 0.5)
    pred = (test["proba"] >= seuil).astype(int)

    k = st.columns(5)
    k[0].metric("ROC AUC", f"{roc_auc_score(y_test, test['proba']):.3f}")
    k[1].metric("PR AUC", f"{average_precision_score(y_test, test['proba']):.3f}")
    k[2].metric("Rappel", f"{recall_score(y_test, pred):.3f}")
    k[3].metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    k[4].metric("Cout metier", f"{cout}", delta=f"{cout - cout_ref:+d} vs seuil 0,50",
                delta_color="inverse")

    g, d = st.columns(2)
    g.markdown("**Matrice de confusion**")
    g.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
                             index=["Reellement sain", "Reellement en defaut"],
                             columns=["Accepte", "Refuse"]), width='stretch')
    g.caption(f"{fn} mauvais payeurs acceptes (cout {bundle['cout_fn']} chacun) · "
              f"{fp} bons clients refuses (cout {bundle['cout_fp']} chacun)")

    seuils = np.linspace(0.01, 0.99, 99)
    courbe = pd.DataFrame({"seuil": seuils,
                           "cout": [cout_metier(y_test, test["proba"], s)[0] for s in seuils]}
                          ).set_index("seuil")
    d.markdown("**Cout total en fonction du seuil**")
    d.line_chart(courbe, color=BLEU, height=260)

    st.subheader("Importance des variables")
    for chemin, legende in [(RACINE / "reports" / "e5_importance.png",
                             "Importance par permutation, mesuree sur le test "
                             "(et non l'importance de Gini, biaisee)")]:
        if chemin.exists():
            st.image(str(chemin), caption=legende, width=750)
        else:
            st.info("Lance `python src/evaluate.py` pour generer les figures.")


# F3  Simulateur

with onglets[2]:
    st.subheader("Simuler une demande de credit")
    brut = charger_brut()
    gabarit = brut.iloc[0].copy()

    with st.form("demande"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Client**")
            age = st.number_input("Age", 18, 78, 38)
            sexe = st.selectbox("Sexe", ["F", "M"])
            situation = st.selectbox("Situation", ["Marie", "Celibataire", "Divorce", "Veuf"])
            personnes = st.number_input("Personnes a charge", 0, 12, 2)
            education = st.selectbox("Education", ["Aucun", "Primaire", "Secondaire", "Superieur"])
            zone = st.selectbox("Zone", ["Urbaine", "Semi-urbaine", "Rurale"])
        with c2:
            st.markdown("**Activite et revenus**")
            secteur = st.selectbox("Secteur", ["Commerce", "Agriculture", "Artisanat",
                                               "Services", "Transport", "Fonction publique"])
            anciennete = st.number_input("Anciennete activite (mois)", 1, 405, 51)
            revenu = st.number_input("Revenu mensuel (FCFA)", 0, 5_000_000, 79_000, step=1_000)
            charges = st.number_input("Charges mensuelles (FCFA)", 0, 5_000_000, 38_400, step=1_000)
            compte = st.selectbox("Compte d'epargne", [0, 1],
                                  format_func=lambda v: "Oui" if v else "Non")
            epargne = st.number_input("Solde d'epargne (FCFA)", 0, 1_000_000, 0, step=1_000)
        with c3:
            st.markdown("**Credit demande**")
            montant = st.number_input("Montant (FCFA)", 25_000, 5_000_000, 289_000, step=1_000)
            duree = st.selectbox("Duree (mois)", [3, 6, 9, 12, 18, 24, 36], index=3)
            taux = st.number_input("Taux annuel (%)", 5.0, 40.0, 21.5, step=0.1)
            objet = st.selectbox("Objet", ["Fonds de roulement", "Equipement", "Intrants agricoles",
                                           "Scolarite", "Habitat", "Sante", "Autre"])
            garantie = st.selectbox("Garantie", ["Caution solidaire", "Aucune", "Materiel", "Hypotheque"])
            groupe = st.selectbox("Groupe solidaire", [0, 1],
                                  format_func=lambda v: "Oui" if v else "Non")

        c4, c5, c6 = st.columns(3)
        score_mm = c4.number_input("Score mobile money (vide = inconnu)", 0.0, 100.0, 54.4)
        score_connu = c4.checkbox("Score mobile money disponible", value=True)
        transactions = c5.number_input("Transactions mobile money / mois", 0, 60, 8)
        distance = c6.number_input("Distance agence (km)", 0.1, 200.0, 5.3)

        envoyer = st.form_submit_button("Evaluer la demande", type="primary")

    if envoyer:
        saisie = {
            "age": age, "sexe": sexe, "situation_matrimoniale": situation,
            "nb_personnes_charge": personnes, "niveau_education": education,
            "zone_habitation": zone, "secteur_activite": secteur,
            "anciennete_activite_mois": anciennete, "revenu_mensuel_fcfa": revenu,
            "charges_mensuelles_fcfa": charges, "possede_compte_epargne": compte,
            "montant_epargne_fcfa": epargne, "montant_credit_fcfa": montant,
            "duree_credit_mois": duree, "taux_interet_annuel": taux,
            "objet_credit": objet, "type_garantie": garantie,
            "membre_groupe_solidaire": groupe, "nb_credits_anterieurs": 2,
            "nb_retards_anterieurs": 0,
            "score_mobile_money": score_mm if score_connu else np.nan,
            "nb_transactions_mm_mois": transactions, "distance_agence_km": distance,
            "date_octroi": pd.Timestamp.today().strftime("%Y-%m-%d"),
        }

        # F4  valeurs hors domaine : on avertit sans bloquer
        alertes = [f"`{c}` = {saisie[c]:,.0f} hors de la plage observee "
                   f"[{lo:,.0f} – {hi:,.0f}] : prediction peu fiable"
                   for c, (lo, hi) in DOMAINE.items()
                   if c in saisie and pd.notna(saisie[c]) and not (lo <= saisie[c] <= hi)]
        if charges >= revenu:
            alertes.append("Les charges depassent le revenu : capacite de remboursement nulle, "
                           "plafonnee a 1 FCFA par le module de preparation.")
        for a in alertes:
            st.warning(a)

        ligne = gabarit.copy()
        for k_, v in saisie.items():
            ligne[k_] = v
        candidat = preparer(pd.DataFrame([ligne]))

        manquantes = [c for c in COLS_MODELE if c not in candidat.columns]
        if manquantes:
            st.error(f"Colonnes absentes apres preparation : {manquantes}")
            st.stop()

        proba = float(modele.predict_proba(candidat[COLS_MODELE])[0, 1])
        refuse = proba >= SEUIL_DEFAUT

        st.divider()
        r1, r2, r3 = st.columns([1, 1, 2])
        r1.metric("Probabilite de defaut", f"{proba*100:.1f} %")
        r2.metric("Seuil de decision", f"{SEUIL_DEFAUT*100:.1f} %")
        if refuse:
            r3.error(f" REFUS\nProbabilite {proba*100:.1f} % ≥ seuil {SEUIL_DEFAUT*100:.1f} %")
        else:
            r3.success(f" ACCORD\nProbabilite {proba*100:.1f} % < seuil {SEUIL_DEFAUT*100:.1f} %")

        st.progress(min(proba / max(SEUIL_DEFAUT * 2, 1e-6), 1.0))
        st.markdown("**Variables metier calculees**")
        cles = ["capacite_remboursement", "mensualite_estimee", "ratio_endettement",
                "credit_sur_revenu", "ratio_charges", "revenu_par_personne"]
        st.dataframe(candidat[cles].T.rename(columns={0: "valeur"}).round(2),
                     width='stretch')
        st.caption("`ratio_endettement` est la variable la plus determinante du modele "
                   "(importance par permutation : 0,285, soit 8 fois la suivante). "
                   "Au-dessus de 1, la mensualite depasse la capacite de remboursement.")
        st.info("Cette estimation assiste la decision, elle ne la remplace pas. "
                "Tout refus doit pouvoir etre reexamine par un agent (cf. section equite).")


# Bonus  scoring par lot

with onglets[3]:
    st.subheader("Scorer un fichier CSV")
    st.caption("Le fichier doit contenir les memes colonnes que "
               "`data/credits_microfinance.csv` (la cible est facultative).")
    fichier = st.file_uploader("Fichier CSV", type="csv")

    if fichier is not None:
        try:
            lot = pd.read_csv(fichier)
        except Exception as e:                                   # F4
            st.error(f"Lecture impossible : {e}")
            st.stop()

        prepare = preparer(lot)
        manquantes = [c for c in COLS_MODELE if c not in prepare.columns]
        if manquantes:
            st.error(f"**{len(manquantes)} colonne(s) manquante(s)** : {manquantes[:10]}")
            st.stop()

        presentes = [c for c in COLS_FUITE if c in lot.columns]
        if presentes:
            st.warning(f"Colonnes post-decision detectees et ignorees : {presentes}")

        prepare["probabilite_defaut"] = modele.predict_proba(prepare[COLS_MODELE])[:, 1]
        prepare["decision"] = np.where(prepare["probabilite_defaut"] >= SEUIL_DEFAUT,
                                       "REFUS", "ACCORD")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dossiers scores", len(prepare))
        c2.metric("Refus", f"{(prepare.decision == 'REFUS').mean()*100:.1f} %")
        c3.metric("Probabilite moyenne", f"{prepare.probabilite_defaut.mean()*100:.1f} %")

        colonnes = ["id_client", "probabilite_defaut", "decision", "ratio_endettement"]
        colonnes = [c for c in colonnes if c in prepare.columns]
        st.dataframe(prepare[colonnes].sort_values("probabilite_defaut", ascending=False),
                     width='stretch', height=320)
        st.download_button("Telecharger les resultats",
                           prepare[colonnes].to_csv(index=False).encode("utf-8"),
                           "resultats_scoring.csv", "text/csv")


# Bonus  suivi d'equite (E6)

with onglets[4]:
    st.subheader("Le modele desavantage-t-il un groupe ?")
    seuil_eq = st.slider("Seuil applique", 0.01, 0.99, float(SEUIL_DEFAUT), 0.01, key="eq")
    test["refuse"] = (test["proba"] >= seuil_eq).astype(int)

    for col in ["sexe", "zone_habitation"]:
        st.markdown(f"**Selon `{col}`**")
        g = test.groupby(col).apply(lambda d: pd.Series({
            "n": len(d),
            "taux de defaut reel": d.defaut_paiement.mean(),
            "taux de refus": d.refuse.mean(),
            "rappel": recall_score(d.defaut_paiement, d.refuse, zero_division=0),
            "precision": precision_score(d.defaut_paiement, d.refuse, zero_division=0),
        }), include_groups=False)
        a, b = st.columns([1, 1])
        a.dataframe(g.round(3), width='stretch')
        b.bar_chart(g[["taux de defaut reel", "taux de refus"]], height=260)
        ecart = g["taux de refus"].max() - g["taux de refus"].min()
        if ecart > 0.05:
            st.warning(
                f"Ecart de **{ecart*100:.1f} points** entre le groupe le plus refuse et le moins "
                f"refuse, alors que les taux de defaut reels s'etendent seulement sur "
                f"{(g['taux de defaut reel'].max()-g['taux de defaut reel'].min())*100:.1f} points. "
                "L'ecart de traitement n'est donc pas justifie par le risque observe.")
