# Dictionnaire des données — `credits_microfinance.csv`

9 255 lignes (dont 52 doublons exacts) × 29 colonnes. Montants en francs CFA (FCFA).
Données **synthétiques** générées par `generateurs/generer_tp1.py`.

## Identifiants et date

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `id_client` | texte | Identifiant du dossier (`CLI-XXXXXX`) | Non unique : doublons présents. À exclure des variables. |
| `date_octroi` | texte | Date d'octroi du crédit (2021-2025) | **Deux formats mélangés** : `AAAA-MM-JJ` et `JJ/MM/AAAA`. Sert à la découpe temporelle. |

## Profil du client

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `age` | entier | Âge en années (18-78) | **96 lignes à `-1`** (sentinelle). |
| `sexe` | catégorie | `F` / `M` | Aussi `Feminin` / `Masculin` (à harmoniser). |
| `situation_matrimoniale` | catégorie | Célibataire, Marié, Divorcé, Veuf | Variantes `MARIE(E)`, `celibataire`. |
| `nb_personnes_charge` | entier | Personnes à charge (0-12) | 1,5 % manquants. |
| `niveau_education` | catégorie | Aucun, Primaire, Secondaire, Supérieur | 2,2 % manquants. Ordinal : envisagez un encodage ordonné. |
| `zone_habitation` | catégorie | Urbaine, Semi-urbaine, Rurale | Variantes en majuscules. |

## Activité et revenus

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `secteur_activite` | catégorie | Commerce, Agriculture, Artisanat, Services, Transport, Fonction publique | Espaces parasites en début de chaîne. |
| `anciennete_activite_mois` | entier | Ancienneté de l'activité (1-405) | 3,7 % manquants. |
| `revenu_mensuel_fcfa` | réel | Revenu mensuel déclaré | **70 lignes à `-999`**, **30 lignes ×1000** (erreur d'unité), 6,1 % manquants. |
| `charges_mensuelles_fcfa` | réel | Charges mensuelles déclarées | — |
| `possede_compte_epargne` | binaire | 1 si compte d'épargne | — |
| `montant_epargne_fcfa` | réel | Solde d'épargne | 9,4 % manquants ; vaut 0 si pas de compte. |

## Caractéristiques du crédit

| Colonne | Type | Description |
|---|---|---|
| `montant_credit_fcfa` | réel | Montant accordé (25 000 - 4 574 000) |
| `duree_credit_mois` | entier | 3, 6, 9, 12, 18, 24 ou 36 mois |
| `taux_interet_annuel` | réel | Taux annuel en % (9 - 37,5) |
| `objet_credit` | catégorie | Fonds de roulement, Équipement, Intrants agricoles, Scolarité, Habitat, Santé, Autre |
| `type_garantie` | catégorie | Aucune, Caution solidaire, Matériel, Hypothèque |
| `membre_groupe_solidaire` | binaire | 1 si le client appartient à un groupe de caution solidaire |

## Historique et empreinte numérique

| Colonne | Type | Description | Pièges |
|---|---|---|---|
| `nb_credits_anterieurs` | entier | Crédits déjà accordés (0-9) | — |
| `nb_retards_anterieurs` | entier | Retards passés (0-5) | Parfois **supérieur** à `nb_credits_anterieurs` : incohérence à corriger. |
| `score_mobile_money` | réel | Score d'activité mobile money (0-100) | **12,8 % manquants** — le manque est-il informatif ? |
| `nb_transactions_mm_mois` | entier | Transactions mobile money par mois | — |
| `distance_agence_km` | réel | Distance à l'agence | **45 lignes à `9999`** (sentinelle). |
| `agent_credit` | catégorie | Agent instructeur (`AG-001` … `AG-026`) | Haute cardinalité. Risque de proxy discriminant : justifiez son usage ou son exclusion. |

## ⛔ Colonnes post-décision — FUITE DE DONNÉES

| Colonne | Type | Pourquoi il faut les supprimer |
|---|---|---|
| `nb_relances_recouvrement` | entier | Le recouvrement n'existe que si le client ne rembourse pas. Corrélation de **0,88** avec la cible. |
| `statut_dossier` | catégorie | Solde, En cours, Contentieux, Recouvrement, Restructuré : les trois dernières valeurs **sont** le défaut. |

## 🎯 Variable cible

| Colonne | Type | Description |
|---|---|---|
| `defaut_paiement` | binaire | **1 = défaut de paiement** (15,5 % des dossiers), 0 = remboursé normalement |

---

## Variables dérivées attendues (partie C)

| Variable | Formule |
|---|---|
| `capacite_remboursement` | `revenu − charges` (plancher à 1) |
| `mensualite_estimee` | `montant × (1 + taux/100 × durée/12) / durée` |
| `ratio_endettement` | `mensualite_estimee / capacite_remboursement` |
| `ratio_charges` | `charges / revenu` |
| `credit_sur_revenu` | `montant_credit / revenu` |
| `taux_retard_historique` | `retards / max(crédits antérieurs, 1)` |
| `revenu_par_personne` | `revenu / (personnes à charge + 1)` |
| `primo_emprunteur` | `1` si `nb_credits_anterieurs == 0` |
