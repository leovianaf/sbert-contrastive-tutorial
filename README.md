# 🧠 Tutorial: Sentence-BERT (sBERT) com Cosine Similarity Loss e Triplet Loss

Este repositório demonstra como treinar e avaliar modelos Sentence-BERT (sBERT) para medir similaridade entre sentenças em português, utilizando duas funções de perda distintas:
- **Cosine Similarity Loss** e **Triplet Loss**, da biblioteca [`sentence-transformers`](https://www.sbert.net/).

---

## 🎯 Objetivos

- Entender como o **sBERT** transforma sentenças em **embeddings**.
- Compreender o papel de uma função de perda (loss function) no treinamento de modelos de machine learning.
- Explorar os conceitos e implementação da **Cosine Similarity Loss** e da **Triplet Loss**.
- Treinar modelos supervisionados para **classificar pares de sentenças** como **semelhantes (1)** ou **diferentes (0)**.
- Avaliar o desempenho dos modelos com métricas como **AUC**, **Accuracy** e **F1-score**.

---

## 📚 Conceitos Fundamentais

### 🔹 O que é Sentence-BERT?

O **Sentence-BERT (sBERT)** é uma variação do modelo BERT ajustada para **comparar sentenças**.
Ele transforma frases em vetores numéricos **(embeddings)** que capturam o seu significado semântico.

Por exemplo:
- *“O gato dorme.”* e *“O felino está descansando.”* terão embeddings **próximos**.
- *“O gato dorme.”* e *“O carro é vermelho.”* terão embeddings **distantes**.

---

## ⚖️ Funções de Perda

### 🔹 O que é uma Função de Perda?

A **função de perda** (ou *loss function*) mede o quanto as previsões do modelo estão **erradas** em relação ao valor **esperado**.
Durante o treinamento, o modelo tenta **minimizar** essa perda, ajustando seus pesos internos a cada iteração.

➡️ Assim, a função de perda é o mecanismo de aprendizado, ela informa ao modelo como melhorar.

### 🔹 Cosine Similarity Loss

A **Cosine Similarity Loss** ensina o modelo a produzir embeddings de sentenças de forma que o cosseno entre eles reflita sua **similaridade semântica**.

- Se duas sentenças têm significados parecidos, o cosseno entre seus embeddings deve ser próximo de 1.
- Se são diferentes, o cosseno deve ser próximo de -1.

### 💡 Fórmula da Cosine Similarity Loss:
$$
L = 1 - \cos(A, B)
$$

Durante o treinamento, o modelo ajusta seus parâmetros para **maximizar a similaridade** entre embeddings de sentenças que são semanticamente próximas e **minimizar a similaridade** entre as que não são.

---

### 🔺 Triplet Loss

A **Triplet Loss** compara três frases por vez:

- **Anchor (A)** – frase base
- **Positive (P)** – frase parecida com A
- **Negative (N)** – frase diferente de A

### 💡 Fórmula da Triplet Loss:
$$
L = \max(0, \; d(A, P) - d(A, N) + \text{margin})
$$

Isso força os embeddings de A e P a ficarem mais **próximos**, e os de A e N, mais **distantes**.

---

### 🧩 Contrastive Loss

A Contrastive Loss é usada quando o dataset contém pares rotulados com 0 ou 1:
- 1 → sentenças semelhantes (devem estar próximas);
- 0 → sentenças diferentes (devem estar distantes).

Em vez de comparar triplas como na Triplet Loss, ela compara apenas pares, penalizando o modelo de acordo com a **distância entre os embeddings** e o **rótulo verdadeiro**.

### 💡 Fórmula da Contrastive Loss:
$$
L = (1 - Y) \cdot D^2 + Y \cdot \max(0, \; m - D)^2
$$

Onde:
- **𝑌** é o rótulo (0 ou 1);
- **𝐷** é a distância entre os embeddings (geralmente Euclidiana);
- **𝑚** é a margem mínima de separação.

---

### 🔶 Multiple Negatives Ranking Loss (MNRL)

A **MultipleNegativesRankingLoss (MNRL)** é uma evolução moderna da *Contrastive Loss*, ela usa o **batch inteiro** como base para aprendizado, sem precisar criar pares ou triplas manualmente.

Durante o treinamento:
- Cada par positivo (A, P) dentro do batch é considerado um exemplo **positivo**;
- Todas as outras sentenças no mesmo batch são tratadas automaticamente como **negativas**.

Assim, o modelo aprende a:
- Maximizar a similaridade entre o par correto (A, P);
- Minimizar a similaridade com todas as outras combinações (A, N).

### 💡 Fórmula da MNRL:
$$
L = -\log \frac{\exp(\cos(A, P))}{\sum_{N \in \text{batch}} \exp(\cos(A, N))}
$$

---

#### 📊 Comparativo geral

| Função de Perda           | Estrutura de entrada       | Ideal para                         | Intuição principal                                    |
| ------------------------- | -------------------------- | ---------------------------------- | ----------------------------------------------------- |
| **CosineSimilarityLoss**  | Pares com score (contínuo) | Similaridade contínua (0–1)/(0-5)  | Aproximar embeddings proporcionalmente ao score       |
| **ContrastiveLoss**       | Pares com label (0/1)      | Classificação binária              | Aproximar pares positivos, afastar negativos          |
| **TripletLoss**           | Triplas (A, P, N)          | Relações relativas entre exemplos  | A é mais parecido com P do que com N                  |
| **MNRL**                  | Pares positivos            | Binário com batches grandes        | Usa todos os pares do batch como negativos implícitos |

---

## 🔧 Pré-requisitos
- Python: 3.11.5 (ou superior)

---

## ⚙️ Instalação e configuração

```bash
git clone https://github.com/leovianaf/sbert-contrastive-tutorial.git
cd sbert-contrastive-tutorial
```

Acessando o repositório, você deve criar seu ambiente virtual, ativá-lo e instalar as dependências:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Com o ambiente virtual configurado, você deve utilizá-lo para executar os notebooks.

---

## 🚀 Treinamento
O notebook de treinamento utilizando a **Cosine Similarity Loss** pode ser acessado aqui: [`train_sbert_cosine.ipynb`](notebooks/train_sbert_cosine.ipynb)

O notebook de treinamento utilizando a **TripletLoss** pode ser acessado aqui: [`train_sbert_triplet.ipynb`](notebooks/train_sbert_triplet.ipynb)

---

## 📈 Avaliação
O notebook de avaliação do treinamento pode ser acessado aqui: [`evaluate_sbert.ipynb`](notebooks/evaluate_sbert.ipynb)

---

## 📈 Métricas de desempenho

| Modelo          | Loss Function        | Accuracy | AUC   | F1-score | Observação                                   |
|-----------------|----------------------|----------|-------|----------|----------------------------------------------|
| sBERT - Cosine  | CosineSimilarityLoss | 0.804    | 0.820 | 0.516    | Modelo com boa separação geral               |
| sBERT - Triplet | TripletLoss          | 0.779    | 0.814 | 0.509    | Boa convergência, mas precisa de refinamento |

---

## 📜 Licença

Este projeto está licenciado sob os termos da **MIT License**.
Você é livre para usar, modificar e distribuir este código, desde que mantenha o aviso de copyright.
