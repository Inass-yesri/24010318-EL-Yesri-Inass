
📄 Compte rendu – Sentiment Analysis sur Actualités Financières (MoE / MoMoE)
=============================================================================
Par El YESRI INASS 
<img width="100" height="150" alt="image" src="https://github.com/user-attachments/assets/8ff73355-eaf0-42d3-ac75-12fdc08df8d2" />

1\. Le Contexte Métier et la Mission
------------------------------------

### Le Problème (Business Case)

Dans la finance, les marchés réagissent en quelques secondes aux **news économiques** :publication de résultats, annonces de banques centrales, fusions, faillites, etc.

Aujourd’hui, les gérants de portefeuille et les traders :

*   lisent manuellement des dizaines de titres par jour,
    
*   évaluent « à la main » si la nouvelle est **positive**, **négative** ou **neutre**,
    
*   risquent des **biais humains** (fatigue, subjectivité, émotions).
    

👉 **Objectif du projet** :Construire un système d’**analyse automatique de sentiment** qui classe chaque titre de news financières en :

*   negative
    
*   neutral
    
*   positive
    

afin d’aider à **prioriser** les informations et, à terme, d’alimenter des modèles de **prédiction de mouvements de marché**.

### L’Enjeu critique

Toutes les erreurs ne se valent pas :

*   Classer une très mauvaise nouvelle en _neutral_ → risque de **sous-réagir** (pertes potentielles).
    
*   Classer une news neutre en _negative_ → risque de **sur-réagir** (ventes inutiles, coûts d’opportunité).
    

🎯 **Enjeu :**Réduire **surtout** les erreurs sur les sentiments _extrêmes_ (très positif / très négatif) qui déclenchent des décisions fortes (achat/vente).

2\. Les Données (L’Input)
-------------------------

La dataset utilisée est le fichier Kaggle all-data (1).csv :

*   **Colonnes principales** :
    
    *   sentiment : negative, neutral, positive
        
    *   text : titre de la news financière
        

Dans le notebook :

```python
df = pd.read_csv("all-data (1).csv", encoding="latin-1")  df.columns = ['sentiment', 'text']
````

### Analyse des données brutes

*   Vérification de la taille : df.shape
    
*   Inspection rapide : df.head(), df.tail()
    
*   Vérification des doublons : df.duplicated().sum()
    
*   Vérification des valeurs manquantes : df.isnull().sum()
    
*   Information sur les types : df.info()
    
*   Nombre de valeurs distinctes : df.nunique()
    

👉 **Constat :**

*   Les textes sont **courts** (titres, pas des articles complets).
    
*   Les labels sentiment sont **déséquilibrés** (une classe plus fréquente, souvent _neutral_).
    
*   Il existe des **doublons** qui sont supprimés avec :
    
```python
df = df.drop_duplicates()
``` `

3\. Le Code Python (Laboratoire)
--------------------------------

Ton notebook joue le rôle de **laboratoire expérimental**. Il enchaîne les grandes phases suivantes :

1.  **Chargement & nettoyage de base** (read\_csv, suppression des doublons, gestion des NaN).
    
2.  **Prétraitement NLP** : fonction clean\_text (lowercase, stopwords, lemmatisation, etc.).
    
3.  **Visualisation simple des classes** : countplot sur sentiment.
    
4.  **Équilibrage des classes** (upsampling avec resample → df\_balanced).
    
5.  **Exploration du vocabulaire** : wordclouds & top words par sentiment.
    
6.  **Représentation des textes** :
    
    *   embeddings de phrase via SentenceTransformer('all-MiniLM-L6-v2') (dimension dense),
        
    *   TF-IDF bigrammes comme autre vue textuelle.
        
7.  **Architecture Mixture-of-Experts (MoE)** :
    
    *   définition d’Expert, SwiGLU, MoEHead,
        
    *   entraînement sur les embeddings.
        
8.  **Agents additionnels** :
    
    *   agent2 = Logistic Regression sur TF-IDF,
        
    *   agent3 = RandomForestClassifier sur TF-IDF.
        
9.  **Meta-Model (MoMoE)** :
    
    *   concaténer les sorties (probabilités) MoE + agent2 + agent3,
        
    *   entraîner un dernier classifieur meta\_clf.
        
10.  **Évaluation** :
    
    *   accuracy\_score, classification\_report,
        
    *   confusion\_matrix → heatmaps pour **MoE** et **MoMoE**.
        

4\. Analyse approfondie : Nettoyage & Prétraitement (Data Wrangling)
--------------------------------------------------------------------

### 4.1. Nettoyage des doublons et valeurs manquantes

Tu supprimes :

*   les lignes dupliquées,
    
*   les éventuelles valeurs manquantes dans le texte :
    

```python
df = df.drop_duplicates()  df.isnull().sum()
````

🧠 **Analyse :**

*   En NLP, des doublons peuvent **biaiser** l’apprentissage : le modèle « revoit deux fois la même phrase », ce qui renforce artificiellement son importance.
    
*   Sur ce projet, supprimer les doublons permet d’obtenir une estimation plus fidèle des performances sur de **nouvelles news**.
    

### 4.2. Fonction de prétraitement clean\_text

La fonction clean\_text (reconstruite à partir de ton notebook) réalise typiquement :

*   Mise en minuscules,
    
*   Suppression des chiffres : re.sub(r'\\d+', '', text)
    
*   Suppression de la ponctuation : re.sub(r'\[^\\w\\s\]', '', text)
    
*   Normalisation des espaces : re.sub(r'\\s+', ' ', text).strip()
    
*   Suppression des stopwords + lemmatisation (ex : organisations → organisation)
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   df['clean_text'] = df['text'].apply(clean_text)  df = df[['clean_text', 'sentiment']]   `

🧠 **Analyse : Pourquoi c’est important ?**

*   Les modèles MoE + logistic regression travaillent sur des **représentations numériques** → bruit lexical inutile = dimension inutile.
    
*   La lemmatisation permet de **regrouper** les formes fléchies d’un même mot (profit / profits / profited…).
    
*   Les stopwords comme the, and, of n’apportent presque aucune information sur le **sentiment** → on gagne en signal/bruit.
    

5\. Analyse approfondie : Équilibrage & Exploration (EDA)
---------------------------------------------------------

### 5.1. Déséquilibre des classes & upsampling

Tu utilises un **ré-échantillonnage par sur-échantillonnage** (upsampling) pour équilibrer les classes :
```python
from sklearn.utils import resample  classes = df['sentiment'].unique()  max_count = df['sentiment'].value_counts().max()  df_list = []  for c in classes:      df_class = df[df['sentiment'] == c]      df_upsampled = resample(df_class,                              replace=True,                              n_samples=max_count,                              random_state=42)      df_list.append(df_upsampled)  df_balanced = pd.concat(df_list)  df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
```

🧠 **Analyse :**

*   Sans équilibrage, le modèle pourrait devenir « paresseux » et prédire **majoritairement la classe dominante** (souvent _neutral_).
    
*   L’upsampling rend les trois sentiments **équitablement représentés** → le modèle est obligé de **se spécialiser** pour reconnaître chaque sentiment.
    

⚠️ **Limite** :L’upsampling répète des phrases déjà vues → risque de **surapprentissage** entendu. Ici, l’utilisation d’un MoE + régularisation et split train/test réduit ce risque, mais c’est un point à surveiller.

### 5.2. Visualisation des sentiments

Avec :

```python
plt.figure(figsize=(6,4))  sns.countplot(x=df['sentiment'])  plt.title("Sentiment Count Plot")
```

puis sur df\_balanced après upsampling.

🧠 **Analyse :**

*   Avant équilibrage : on voit le **déséquilibre** (ex : plus de neutral).
    
*   Après équilibrage : les trois barres sont de **hauteur similaire** → condition nécessaire pour entraîner un meta-modèle robuste qui ne néglige pas les sentiments minoritaires.
    

### 5.3. WordClouds & Top Words

Tu génères :

*   un wordcloud global (all\_text = " ".join(df\['clean\_text'\])),
    
*   des wordclouds par sentiment (for c in unique\_classes),
    
*   un graphique des **Top N mots par sentiment** :
    

```python
from collections import Counter  def plot_top_words(df, sentiment_col='sentiment', text_col='clean_text', top_n=10):      sentiments = df[sentiment_col].unique()      for s in sentiments:          texts = df[df[sentiment_col] == s][text_col]          all_words = " ".join(texts).split()          word_counts = Counter(all_words).most_common(top_n)          ...
```

🧠 **Analyse :**

*   Les wordclouds permettent d’identifier les **mots typiques** :
    
    *   positive : _growth, profit, upbeat, surge, beat estimates…_
        
    *   negative : _loss, drop, downgrade, miss, slump, crisis…_
        
    *   neutral : _announces, reports, said, plans…_
        
*   Ça valide l’hypothèse que le sentiment est **corrélé au vocabulaire**, ce qui justifie une approche machine learning supervisée.
    

6\. Analyse approfondie : Méthodologie (Split & Représentation)
---------------------------------------------------------------

### 6.1. Encodage des labels

Tu crées une version numérique du sentiment :

```python
mapping = {'positive': 1, 'neutral': 0, 'negative': -1}  df_balanced['sentiment_num'] = df_balanced['sentiment'].map(mapping)
```

🧠 **Analyse :**

*   Cet encodage garde la **structure ordinale** : -1 < 0 < 1 (négatif → neutre → positif), ce qui est cohérent avec une lecture de _score_.
    
*   Plus tard, tu reviendras à des labels humains via map({0:-1,1:0,2:1}) pour interpréter les prédictions.
    

### 6.2. Représentation des textes

Tu utilises deux familles de représentations :

#### a) Sentence Transformer (Embeddings denses)
```python
from sentence_transformers import SentenceTransformer  model = SentenceTransformer('all-MiniLM-L6-v2')  embeddings = model.encode(df_balanced["clean_text"].tolist(), show_progress_bar=True)
```

*   Chaque phrase → vecteur dense (par ex. dimension 384).
    
*   Ces embeddings capturent **le contexte** (proche de BERT, adapté aux phrases).
    

🧠 **Analyse :**

*   Beaucoup plus **riches** sémantiquement que du simple bag-of-words.
    
*   Parfait pour être injectés dans un réseau type **MoE**.
    

#### b) TF-IDF (N-grammes)

```python
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))  tfidf_fit_texts = df_balanced["clean_text"].astype(str).tolist()  tfidf_matrix = tfidf.fit_transform(tfidf_fit_texts)  X_train_tfidf = tfidf_matrix[idx_train]  X_test_tfidf = tfidf_matrix[idx_test]   `
```

🧠 **Analyse :**

*   TF-IDF nourrit les modèles **linéaires** (Logistic Regression) et **ensemble** (RandomForest).
    
*   Les bigrammes (1,2) permettent de capturer des expressions comme _"beats estimates"_ ou _"misses expectations"_, très importantes en finance.
    

### 6.3. Split train/test & reproductibilité

Tu crées des indices de train/test :

  ```python
  idx_train, idx_test, y_train, y_test = train_test_split(      np.arange(len(df_balanced)),       df_balanced["sentiment_num"].values,       test_size=0.2,       random_state=42,       stratify=df_balanced["sentiment_num"].values  )
```

🧠 **Analyse :**

*   test\_size=0.2 → 80% pour l’apprentissage, 20% pour l’évaluation.
    
*   stratify → la distribution des classes est **la même** dans train et test.
    
*   random\_state=42 → reproductibilité scientifique : tout le monde obtient **exactement** le même split.
    

7\. FOCUS THÉORIQUE : L’Architecture Mixture-of-Experts (MoE) & Meta-Model
--------------------------------------------------------------------------

Pourquoi un simple modèle ne suffit pas ici ?

### 7.1. Faiblesse d’un seul « expert »

Un seul modèle (ex : Logistic Regression ou Random Forest) :

*   peut être très bon sur certains types de phrases (ex : titres simples),
    
*   mais moins performant sur des formulations plus complexes, ironiques, ou très techniques.
    

On veut **plusieurs points de vue** :

*   un modèle qui sait bien gérer la structure sémantique (embeddings + MoE),
    
*   un modèle qui sait bien exploiter des n-grammes fréquents (TF-IDF + LR),
    
*   un modèle capable de capter des interactions non-linéaires (TF-IDF + RandomForest).
    

### 7.2. Architecture MoE (sur Sentence Embeddings)

Dans ton code, tu définis :

```python
class SwiGLU(nn.Module):      ...  class Expert(nn.Module):      def __init__(self, dim, hidden_mult=4):          ...      def forward(self, x):          ...  class MoEHead(nn.Module):      def __init__(self, dim, num_experts=4, k=2, num_classes=3):          ...      def forward(self, x):          gate_logits = self.router(x)          gate_probs = F.softmax(gate_logits, dim=-1)          # top-k experts activés pour chaque sample          ...          return logits, gate_probs   `
```

🧠 **Analyse :**

*   **Experts** : chaque Expert est un petit réseau qui apprend une **spécialisation** sur certains types de phrases.
    
*   **Router** : pour chaque phrase, le router calcule une distribution de probabilité sur les experts → il choisit les k meilleurs (top-k).
    
*   L’output final est un **mélange pondéré** des sorties des experts, suivi d’un classifier qui donne une probabilité pour chaque sentiment.
    

Intuition :

> Chaque titre de news est envoyé principalement à **2 experts** parmi 4, ceux qui sont « le plus compétents » pour ce cas particulier.

### 7.3. Agents classiques : Logistic Regression & Random Forest

```python
  agent2 = LogisticRegression(max_iter=1000)  agent2.fit(X_train_tfidf, np.array(y_train))  agent2_proba = agent2.predict_proba(X_test_tfidf)  agent3 = RandomForestClassifier(n_estimators=200, random_state=42)  agent3.fit(X_train_tfidf, np.array(y_train))  agent3_proba = agent3.predict_proba(X_test_tfidf)
```

🧠 **Analyse :**

*   **Agent 2 (LR)** : stable, interprétable, très adapté pour TF-IDF (espace à haute dimension mais linéairement séparables).
    
*   **Agent 3 (RF)** : capte des interactions non-linéaires entre n-grammes, robustes au bruit, mais moins interprétable.
    

### 7.4. Meta-Model (MoMoE) : combiner les cerveaux

Tu construis un **meta-dataset** :

*   features = concaténation des proba/sorties de :
    
    *   MoE (moe\_test\_proba)
        
    *   agent2 (agent2\_proba)
        
    *   agent3 (agent3\_proba)
        
*   labels = y\_test
    

Puis tu entraînes un **meta classifieur** (par ex. Logistic Regression ou autre) :

```python
meta_X = np.concatenate([moe_test_proba, agent2_proba, agent3_proba], axis=1)  meta_y = y_test  meta_clf = LogisticRegression(max_iter=1000)  meta_clf.fit(meta_X, meta_y)  meta_preds = meta_clf.predict(meta_X)
```

🧠 **Analyse :**

*   Le meta-model apprend **quand faire confiance à quel expert** :
    
    *   certaines zones de l’espace des phrases → MoE est meilleur,
        
    *   d’autres → l’agent TF-IDF est plus fiable.
        
*   C’est une **deuxième couche d’intelligence** qui orchestre les prédictions.
    

8\. Analyse approfondie : Évaluation (l’Heure de vérité)
--------------------------------------------------------

### 8.1. Performance du MoE seul

Tu calcules :

```python
 print("MoE test accuracy:", accuracy_score(y_test, all_preds_moe))  print(classification_report(y_test, all_preds_moe, digits=4))
```

Et visualises la matrice de confusion :

```python
cm_moe = confusion_matrix(y_test, all_preds_moe)  sns.heatmap(cm_moe, annot=True, fmt='d', cmap='Blues')  plt.title("Confusion Matrix - MoE")
```

🧠 **Analyse qualitative (sans chiffres exacts)** :

*   MoE se base sur des **embeddings sémantiques** → très bon pour :
    
    *   reconnaître des sentiments exprimés de manière subtile,
        
    *   généraliser à des formulations nouvelles.
        
*   Les erreurs typiques :
    
    *   _neutral_ ↔ _positive_ quand le texte est vaguement optimiste,
        
    *   _neutral_ ↔ _negative_ quand le titre annonce un risque sans impact immédiat.
        

### 8.2. Performance du Meta-Model (MoMoE)

Tu calcules :
```python
print("MoMoE (meta) accuracy:", accuracy_score(meta_y, meta_preds))  print(classification_report(meta_y, meta_preds, digits=4))
```

Et la matrice de confusion :

```python
cm_momoe = confusion_matrix(meta_y, meta_preds)  sns.heatmap(cm_momoe, annot=True, fmt='d', cmap='Greens')  plt.title("Confusion Matrix - MoMoE (Meta Model)")
```

🧠 **Analyse comparative MoE vs MoMoE :**

Même sans mettre de chiffres précis, on peut analyser la **tendance attendue** :

1.  **Accuracy globale**
    
    *   MoMoE devrait **au moins égaler**, et souvent **légèrement dépasser** le MoE seul, car il exploite **trois sources d’information** au lieu d’une.
        
2.  **Par classe (Précision / Rappel)**
    
    *   Sur la classe negative :
        
        *   logistic regression + TF-IDF est souvent très fort (mots comme _loss, slump, downgrade, miss_).
            
        *   MoMoE apprend à peser davantage cet agent sur ce type de phrases.
            
    *   Sur la classe positive :
        
        *   même logique avec des bigrammes comme _beats estimates_, _raises guidance_.
            
    *   Sur la classe neutral :
        
        *   plus ambiguë → le MoE (embeddings contextuels) a une vraie valeur ajoutée.
            
3.  **Matrice de confusion**
    
    *   Les erreurs les plus **critiques** du point de vue métier :
        
        *   negative → neutral ou positive (sous-estimer une mauvaise nouvelle),
            
        *   positive → neutral (ne pas voir une opportunité).
            
    *   MoMoE devrait **réduire** ces erreurs par rapport au MoE seul, car il combine plusieurs regards.
        

### 8.3. Interprétation métier des métriques

En pratique, on regarde particulièrement :

*   **Recall sur negative** :« Parmi toutes les véritables mauvaises news, combien mon système en détecte-t-il ? »
    
*   **Precision sur positive** :« Quand il dit que c’est positif, a-t-on vraiment une bonne nouvelle, ou est-il trop optimiste ? »
    
*   **F1-score** par classe : compromis global entre Precision et Recall.
    

👉 **Lien avec la finance :**

*   Un high Recall sur negative permet d’éviter les **surprises catastrophiques** (crash, pertes, faillites).
    
*   Un modèle légèrement « paranoïaque » (prédit plus souvent negative que nécessaire) peut être acceptable selon la **tolérance au risque du métier**.
    

9\. Conclusion du Projet
------------------------

Ce projet montre comment passer d’un **CSV brut de titres financiers** à un **système avancé de classification de sentiment** basé sur :

*   un prétraitement NLP solide (cleaning, lemmatisation, stopwords),
    
*   une gestion intelligente du **déséquilibre de classes** (upsampling),
    
*   des **représentations hybrides** (embeddings + TF-IDF),
    
*   une architecture **Mixture-of-Experts** (MoE) enrichie d’un **meta-model** (MoMoE).
    

💡 **Apport principal de l’approche MoE / MoMoE :**

*   Au lieu de chercher **un seul modèle parfait**, tu combines plusieurs modèles qui se **complètent**, avec un meta-modèle qui apprend _quand_ faire confiance à quel expert.
    

📌 **Perspectives d’amélioration possibles :**

*   Utiliser un modèle pré-entraîné spécialisé finance : **FinBERT**.
    
*   Ajouter une dimension **temps** : relier chaque news au mouvement réel du marché (backtesting).
    
*   Définir une **matrice de coûts** métier (erreur sur negative plus pénalisante que sur neutral) et adapter la fonction de perte.
    
