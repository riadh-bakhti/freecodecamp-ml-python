# Certification freeCodeCamp — Machine Learning with Python

Solutions complètes des **5 projets** de la certification
[Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python/)
de freeCodeCamp.

| # | Projet | Fichier | Résultat | Seuil |
|---|--------|---------|----------|-------|
| 1 | Rock Paper Scissors | `RPS.ipynb` (et `RPS.py`) | 84,9 % à 100 % contre les 4 bots | ≥ 60 % |
| 2 | Cat and Dog Image Classifier | `fcc_cat_dog.ipynb` | 96 % sur les 50 images de test | ≥ 63 % |
| 3 | Book Recommendation Engine (KNN) | `fcc_book_recommendation_knn.ipynb` | Test réussi (distances 0,802 / 0,771) | — |
| 4 | Linear Regression Health Costs | `fcc_predict_health_costs_with_regression.ipynb` | MAE ≈ 2 367 | < 3 500 |
| 5 | Neural Network SMS Text Classifier | `fcc_sms_text_classification.ipynb` | 7/7 messages, 98,3 % de validation | 7/7 |

Les notebooks contiennent déjà les **sorties d'exécution** (résultats, graphiques)
et les cellules de test officielles de freeCodeCamp.

---

## 1. Rock Paper Scissors — `RPS.ipynb`

Le joueur utilise un **ensemble de prédicteurs** du prochain coup adverse :

- chaînes de Markov d'ordre 1 à 5 sur l'historique de l'adversaire ;
- modèles des trois bots adaptatifs, qui réagissent à *notre* historique :
  - `kris` contre notre dernier coup,
  - `mrugesh` contre notre coup le plus fréquent sur les 10 derniers,
  - `abbey` contre sa prédiction d'ordre 2 de notre prochain coup.

Chaque prédicteur est noté sur les 80 dernières parties ; le meilleur est utilisé
et on joue le coup qui le bat.

Résultats obtenus (1 000 parties par bot) :

| Bot | Taux de victoire |
|-----|------------------|
| quincy | 99,8 % |
| abbey | 84,9 % |
| kris | 100,0 % |
| mrugesh | 99,4 % |

Le fichier `RPS.py` contient la même solution, testable avec `python3 -m unittest test_module -v`.

## 2. Cat and Dog Image Classifier — `fcc_cat_dog.ipynb`

- Base **MobileNetV2** pré-entraînée sur ImageNet (`include_top=False`), gelée,
  surmontée d'un `GlobalAveragePooling2D`, d'un `Dropout` et d'une couche de sortie.
- Étape 1 : entraînement de la tête de classification (8 époques, augmentation
  des données : rotations, translations, zoom, cisaillement, retournement).
- Étape 2 : fine-tuning des 30 dernières couches de MobileNetV2 (8 époques, LR 1e-5).
- Évaluation : moyenne des prédictions de l'image et de son miroir horizontal
  (TTA), ce qui stabilise le score.

Le transfert d'apprentissage est utilisé car il est nettement plus fiable qu'un
CNN entraîné de zéro sur les 2 000 images d'entraînement du jeu de données
freeCodeCamp (un CNN « maison » plafonnait autour de 62 % sur le jeu de test).

## 3. Book Recommendation Engine (KNN) — `fcc_book_recommendation_knn.ipynb`

- Filtrage : utilisateurs ayant au moins 200 notes, livres ayant au moins
  100 notes (comptages calculés sur le jeu de données d'origine).
- Fusion livres/notes, suppression des doublons `(utilisateur, titre)`.
- Table pivot `titre × utilisateur`, matrice creuse `csr_matrix`.
- `NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')`.
- `get_recommends(book)` renvoie les 5 livres les plus proches, du plus éloigné
  au plus proche, comme attendu par le test.

## 4. Linear Regression Health Costs — `fcc_predict_health_costs_with_regression.ipynb`

- Encodage one-hot de `sex`, `smoker`, `region`.
- Séparation 80/20, normalisation adaptée sur l'ensemble d'entraînement.
- Réseau `Dense(64, relu) → Dense(64, relu) → Dense(1)`, perte MAE,
  120 époques.
- **MAE sur le jeu de test : ≈ 2 367** (seuil : 3 500).

## 5. Neural Network SMS Text Classifier — `fcc_sms_text_classification.ipynb`

- `TextVectorization` (10 000 tokens, séquences de 120).
- `Embedding(10000, 32)` → `Bidirectional(LSTM(32))` → `Dense(32, relu)` →
  `Dropout(0.3)` → `Dense(1, sigmoid)`.
- 12 époques, précision de validation ≈ **98,3 %**.
- `predict_message()` renvoie `[probabilité, 'spam' | 'ham']` ; les 7 messages
  du test sont correctement classés.

---

## Exécution locale

```bash
pip install -r requirements.txt
jupyter notebook
```

Les jeux de données sont dans `data/` ; les notebooks les téléchargent
automatiquement s'ils sont absents.

Pour exécuter tous les notebooks en ligne de commande :

```bash
jupyter nbconvert --to notebook --execute --inplace RPS.ipynb
jupyter nbconvert --to notebook --execute --inplace fcc_cat_dog.ipynb
jupyter nbconvert --to notebook --execute --inplace fcc_book_recommendation_knn.ipynb
jupyter nbconvert --to notebook --execute --inplace fcc_predict_health_costs_with_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace fcc_sms_text_classification.ipynb
```

## Soumission sur freeCodeCamp

freeCodeCamp demande un **lien Google Colab** pour chaque projet :

1. Ouvrir <https://colab.research.google.com/> → *Importer* → choisir le
   notebook `.ipynb` correspondant.
2. Menu *Exécution* → **Tout exécuter** (les données se téléchargent toutes
   seules ; l'entraînement chats/chiens prend quelques minutes).
3. Vérifier que la dernière cellule affiche bien « You passed the challenge! ».
4. *Partager* → « Toute personne disposant du lien » → copier le lien.
5. Sur la page du projet freeCodeCamp, coller le lien et valider.

> La certification est liée au compte freeCodeCamp : la soumission finale doit
> être faite depuis ce compte.

## Remarques techniques

- Environnement de développement : Linux ARM64 (aarch64), Python 3.12.
  TensorFlow n'a pas de wheel officielle PyPI pour cette plateforme ;
  le build communautaire `tensorflow-aarch64==2.16.1` a été utilisé
  localement. Sur Colab (x86_64), `tensorflow` standard fonctionne.
- `numpy` doit rester en version `< 2` avec ce build de TensorFlow.
- Les notebooks restent compatibles avec TensorFlow/Keras 2.x et 3.x.
