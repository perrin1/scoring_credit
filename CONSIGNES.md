# TP 1 — Scoring de crédit en microfinance

**Domaine :** finance inclusive · **Tâche :** classification binaire déséquilibrée
**Données :** `credits_microfinance.csv` — 9 255 lignes × 29 colonnes
**Durée estimée :** 8 à 12 h · **Livrable :** dépôt GitHub + app Streamlit

---

## 1. Contexte

Une institution de microfinance accorde des crédits à des commerçants, artisans et
producteurs agricoles. Elle vous transmet l'historique de **9 255 dossiers** octroyés
entre 2021 et 2025, avec pour chacun l'issue du remboursement.

Objectif : construire un modèle qui estime, **au moment de la demande**, la probabilité
qu'un dossier finisse en défaut de paiement — et le rendre exploitable par un agent de
crédit via une interface Streamlit.

Le taux de défaut observé est d'environ **15,5 %** : les classes sont déséquilibrées,
l'*accuracy* n'est donc pas une métrique acceptable.

> ⚠️ Les données sont **synthétiques**. Les relations qu'elles contiennent sont
> réalistes mais aucune conclusion réelle ne doit en être tirée.

---

## 2. Objectifs pédagogiques

À la fin du TP, vous devez savoir :

1. auditer et nettoyer un jeu de données sale (manquants, sentinelles, doublons,
   incohérences de saisie, erreurs d'unité) ;
2. **identifier et éliminer une fuite de données** (*data leakage*) ;
3. construire des variables métier plus prédictives que les colonnes brutes ;
4. mettre en place un pipeline `scikit-learn` reproductible (`ColumnTransformer` +
   `Pipeline`) sans fuite entre entraînement et validation ;
5. traiter un déséquilibre de classes et **choisir un seuil de décision par le coût
   métier** plutôt que d'utiliser 0,5 par défaut ;
6. évaluer avec les bonnes métriques (ROC AUC, PR AUC, rappel, calibration) ;
7. interpréter le modèle et interroger son équité ;
8. livrer le tout dans un dépôt GitHub propre avec une démo Streamlit.

---

## 3. Le piège central du TP

Deux colonnes du fichier ne sont connues **qu'après** le défaut de paiement :

| Colonne | Pourquoi c'est une fuite |
|---|---|
| `nb_relances_recouvrement` | Les relances de recouvrement n'existent que si le client ne paie pas |
| `statut_dossier` | Contient `Contentieux` / `Recouvrement` : c'est la cible déguisée |

Les conserver donne une **ROC AUC de 1,00** en test et un modèle **totalement
inutilisable en production**, puisque ces informations n'existent pas au moment de la
décision.

**Expérience demandée** (à documenter dans votre rendu) : entraînez le même modèle deux
fois, une fois avec ces deux colonnes et une fois sans, puis comparez les AUC. Vous devez
obtenir environ 1,00 contre 0,82. Cette comparaison vaut des points : elle prouve que
vous avez compris le mécanisme, pas seulement suivi une consigne.

**Toute soumission dont le modèle final utilise ces colonnes est notée 0 sur la partie
modélisation.**

---

## 4. Consignes

### Partie A — Audit et nettoyage (4 points)

- A1. Chargez le fichier brut. Produisez un tableau du taux de valeurs manquantes par
  colonne et commentez : les manquants sont-ils aléatoires ou liés à un profil de client ?
  *(Indice : `score_mobile_money` manque à 12,8 % — qui sont ces clients ?)*
- A2. Détectez les **doublons exacts** et traitez-les.
- A3. Trouvez les **valeurs sentinelles** encodées en numérique (au moins trois :
  `age = -1`, `revenu_mensuel_fcfa = -999`, `distance_agence_km = 9999`) et convertissez-les
  en `NaN`. Justifiez pourquoi il est dangereux de les laisser.
- A4. Harmonisez les colonnes catégorielles : casse, espaces parasites, synonymes
  (`Feminin`/`F`, `MARIE(E)`/`Marie`, `SEMI-URBAINE`/`Semi-urbaine`…).
- A5. La colonne `date_octroi` mélange deux formats (`AAAA-MM-JJ` et `JJ/MM/AAAA`).
  Parsez-la correctement et extrayez année / mois / trimestre.
- A6. Repérez les **erreurs d'unité** sur `revenu_mensuel_fcfa` (30 lignes saisies ×1000)
  et corrigez-les. Traitez les incohérences métier (`nb_retards_anterieurs >
  nb_credits_anterieurs`, épargne négative…).
- A7. Choisissez une stratégie pour les valeurs extrêmes (winsorisation, log, suppression)
  et **justifiez-la**.

### Partie B — Analyse exploratoire (3 points)

- B1. Taux de défaut par `secteur_activite`, `type_garantie`, `zone_habitation`,
  `duree_credit_mois`. Quels segments sont les plus risqués ?
- B2. Distributions comparées (défaut / sain) de 4 variables numériques au choix.
- B3. Matrice de corrélation. **Quelle colonne est anormalement corrélée à la cible, et
  pourquoi devez-vous la supprimer ?**
- B4. Le `score_mobile_money` apporte-t-il de l'information au-delà du revenu ? Vérifiez.
- B5. Trois graphiques minimum, chacun accompagné d'une phrase d'interprétation.

### Partie C — Ingénierie de variables (3 points)

Créez au moins **six** variables dérivées, dont obligatoirement :

- `capacite_remboursement` = revenu − charges (plancher à 1) ;
- `mensualite_estimee` = montant × (1 + taux/100 × durée/12) / durée ;
- `ratio_endettement` = mensualité / capacité de remboursement ← **la plus discriminante** ;
- `taux_retard_historique` = retards / crédits antérieurs ;
- `credit_sur_revenu`, `revenu_par_personne`, `epargne_sur_credit`, `primo_emprunteur`…

Ajoutez au moins **une variable de votre invention** et démontrez son apport (avec /
sans, à modèle égal). Testez aussi un indicateur binaire « valeur manquante » : le fait
de ne pas avoir de score mobile money est-il en soi informatif ?

### Partie D — Modélisation (5 points)

- D1. **Séparation temporelle** : triez par `date_octroi`, entraînez sur les 80 % les
  plus anciens et testez sur les 20 % les plus récents. Expliquez pourquoi un
  `train_test_split` aléatoire est optimiste dans un contexte de scoring.
- D2. Construisez un `Pipeline` = `ColumnTransformer` (imputation + encodage) + estimateur.
  **L'imputation doit être apprise dans le pipeline**, jamais sur le jeu complet avant
  la découpe.
- D3. Entraînez et comparez au moins **trois familles de modèles** :
  régression logistique (référence interprétable), forêt aléatoire, *gradient boosting*
  (`HistGradientBoostingClassifier`, ou XGBoost / LightGBM en bonus).
- D4. Validation croisée stratifiée 5 blocs sur l'entraînement, scoring `roc_auc`.
  Rapportez moyenne **et écart-type**.
- D5. Traitez le déséquilibre — au moins **deux** approches comparées :
  `class_weight`, `scale_pos_weight`, sous-échantillonnage, SMOTE (`imbalanced-learn`).
- D6. Réglez les hyperparamètres du meilleur modèle (`GridSearchCV` ou
  `RandomizedSearchCV`) et documentez la grille explorée.

### Partie E — Évaluation et décision (4 points)

- E1. Sur le jeu de test : ROC AUC, PR AUC, précision, rappel, F1, matrice de confusion.
- E2. **Pourquoi la PR AUC est-elle plus informative que la ROC AUC ici ?**
- E3. Optimisez le **seuil de décision** avec la matrice de coûts fournie
  (faux négatif = 5, faux positif = 1) : un mauvais payeur accepté coûte cinq fois plus
  cher qu'un bon client refusé. Tracez le coût total en fonction du seuil et justifiez
  votre choix. Comparez au seuil 0,5.
- E4. **Calibration** : le modèle annonce-t-il des probabilités fiables ?
  (`calibration_curve`, score de Brier ; testez `CalibratedClassifierCV`.)
- E5. **Interprétabilité** : importance par permutation (pas l'importance Gini brute,
  biaisée vers les variables à forte cardinalité) et, en bonus, SHAP.
- E6. **Équité** : comparez le taux de refus et le rappel selon `sexe` et
  `zone_habitation`. Le modèle désavantage-t-il un groupe ? Que proposez-vous ?

### Partie F — Application Streamlit (3 points)

Construisez un tableau de bord qui rend votre modèle utilisable par un agent de crédit.
Sérialisez votre pipeline entraîné (`joblib.dump`) et chargez-le dans l'app avec
`@st.cache_resource` plutôt que de réentraîner à chaque lancement.

Attendu au minimum :

- F1. un onglet **exploration** : taux de défaut par segment, distributions ;
- F2. un onglet **performance** : vos métriques, matrice de confusion, importance des
  variables ;
- F3. un **simulateur** : formulaire de demande de crédit renvoyant une probabilité de
  défaut et une décision selon votre seuil ;
- F4. la gestion des cas limites (modèle absent, valeur hors domaine, colonne manquante).

Bonus valorisés : scoring par lot depuis un CSV téléversé avec export des résultats,
curseur de seuil interactif montrant l'impact sur le coût métier, explication SHAP
individuelle, onglet de suivi d'équité.

⚠️ Piège classique : reconstruire les variables dérivées différemment dans l'app et dans
l'entraînement. Mettez votre code de préparation dans **un seul module** importé par les
deux.

### Partie G — Livraison GitHub (3 points)

- G1. Dépôt public avec une arborescence claire (`data/`, `src/`, `app/`, `notebooks/`,
  `models/`, `reports/`).
- G2. `README.md` contenant : le problème, la description des données, votre démarche,
  **un tableau de résultats**, une capture d'écran de l'app, les instructions
  d'installation et de lancement, les limites du travail.
- G3. `requirements.txt` avec versions, `.gitignore`, commits atomiques en français ou
  anglais (pas de `update`, `fix2`, `final_final`).
- G4. Code reproductible : `random_state` fixé partout, exécution de bout en bout
  possible depuis un clone propre.
- G5. Section **« Limites et éthique »** : que se passe-t-il si ce score est déployé ?
  Quels biais, quel recours pour le client refusé, quelle supervision humaine ?

---

## 5. Barème (/25, ramené sur 20)

| Partie | Points |
|---|---|
| A — Audit et nettoyage | 4 |
| B — Analyse exploratoire | 3 |
| C — Ingénierie de variables | 3 |
| D — Modélisation | 5 |
| E — Évaluation et décision | 4 |
| F — Application Streamlit | 3 |
| G — Livraison GitHub | 3 |
| **Total** | **25** |

**Pénalités :** utilisation des colonnes de fuite → 0 à la partie D.
Imputation ou standardisation calculée avant la découpe train/test → −3.
Absence de séparation temporelle → −2. Métrique unique = accuracy → −2.

---

## 6. Performances de référence

Ces valeurs ont été mesurées sur ce jeu de données lors de sa conception : séparation
temporelle 80/20, colonnes de fuite retirées, imputation et encodage one-hot appris
**dans** le pipeline, `class_weight="balanced"`, validation croisée stratifiée à 5 blocs.
Elles vous donnent un ordre de grandeur — si vous en êtes très loin dans un sens ou dans
l'autre, cherchez l'erreur.

| Modèle | CV ROC AUC | Test ROC AUC | Test PR AUC | Rappel @ seuil coût |
|---|---|---|---|---|
| Régression logistique | 0,844 ± 0,010 | **0,825** | 0,471 | 0,807 |
| Forêt aléatoire | 0,840 ± 0,008 | 0,812 | 0,459 | 0,740 |
| HistGradientBoosting | 0,839 ± 0,007 | 0,820 | **0,490** | 0,698 |

**Votre objectif : dépasser 0,84 de ROC AUC en test sans fuite de données.**
Un score supérieur à 0,95 est le signe presque certain d'une fuite — cherchez l'erreur.

Que le modèle linéaire tienne tête au *boosting* est un résultat intéressant à
commenter : le signal principal (`ratio_endettement`) est largement monotone.
Trouvez où les modèles non linéaires peuvent réellement gagner
(interactions `secteur_activite × duree_credit_mois`, effets d'âge non monotones…).

---

## 7. Livrables attendus

| Fichier | Contenu |
|---|---|
| `notebooks/01_exploration.ipynb` | Parties A et B : audit, nettoyage, graphiques commentés |
| `src/preprocessing.py` | Votre nettoyage et vos variables dérivées, réutilisable |
| `src/train.py` | Entraînement, validation croisée, comparaison, sérialisation |
| `src/evaluate.py` | Métriques, figures, choix du seuil |
| `app/streamlit_app.py` | Le tableau de bord |
| `models/` | Votre pipeline sérialisé (`joblib`) |
| `README.md` | Problème, démarche, tableau de résultats, capture d'écran, limites |
| `requirements.txt` | Versions figées |

Environnement minimal : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`,
`streamlit`. Optionnels selon vos bonus : `xgboost`, `lightgbm`, `imbalanced-learn`, `shap`.

**Conseil de méthode** : faites les parties A à C dans un notebook, puis **sortez le code
qui marche dans des fichiers `.py`**. Un projet entièrement contenu dans un notebook est
difficile à relire et impossible à réutiliser — c'est pénalisé en partie G.

## 8. Bonus (+2 max)

- XGBoost ou LightGBM avec arrêt anticipé, comparé proprement à la baseline.
- Empilement (*stacking*) de modèles.
- SHAP : explication globale + explication individuelle intégrée à l'app.
- Analyse de **dérive temporelle** : la performance se dégrade-t-elle sur les
  dossiers les plus récents ? Tracez l'AUC par trimestre d'octroi.
- Suivi d'expériences avec MLflow.
- API FastAPI + `Dockerfile` en plus de l'app Streamlit.
