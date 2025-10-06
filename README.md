# 🧠 Tutorial: Sentence-BERT (sBERT) com Cosine Similarity Loss

Este repositório demonstra **como treinar um modelo Sentence-BERT (sBERT)** para medir **similaridade entre sentenças** utilizando a função de perda **ContrastiveLoss** da biblioteca [`sentence-transformers`](https://www.sbert.net/).

---

## 🎯 Objetivos

- Entender como o **sBERT** gera embeddings semânticos de sentenças.
- Compreender o papel de uma função de perda (loss function) no treinamento de modelos de machine learning.
- Aprender o conceito e implementação da **Cosine Similarity Loss**.
- Treinar um modelo supervisionado para prever o grau de **similaridade** entre pares de sentenças (0 = diferentes, 5 = muito semelhantes).
- Avaliar o modelo usando **Cosine Similarity** e **Mean Squared Error (MSE)**.

---

## 📚 Conceitos Fundamentais

### 🔹 O que é Sentence-BERT?

O **Sentence-BERT (sBERT)** é uma variação do modelo BERT ajustada para **comparar sentenças**.
Ele transforma frases em vetores numéricos **(embeddings)** que capturam o seu significado semântico.

Por exemplo:
- *“O gato dorme.”* e *“O felino está descansando.”* terão embeddings **próximos**.
- *“O gato dorme.”* e *“O carro é vermelho.”* terão embeddings **distantes**.

### 🔹 O que é uma Função de Perda?

A **função de perda** (ou *loss function*) mede o quanto as previsões do modelo estão **erradas** em relação ao valor **esperado**.
Durante o treinamento, o modelo tenta **minimizar** essa perda, ajustando seus pesos internos a cada iteração.

➡️ Assim, a função de perda é o mecanismo de aprendizado, ela informa ao modelo como melhorar.

### 🔹 O que é a Cosine Similarity Loss?

A **Cosine Similarity Loss** ensina o modelo a produzir embeddings de sentenças de forma que o cosseno entre eles reflita sua **similaridade semântica**.

- Se duas sentenças têm significados parecidos, o cosseno entre seus embeddings deve ser próximo de 1.
- Se são diferentes, o cosseno deve ser próximo de -1.

#### 💡 Fórmula da Similaridade do Cosseno:
**cos_sim(A, B) = (A · B) / (‖A‖ × ‖B‖)**

### 💡 Fórmula da Cosine Similarity Loss:
$$
L = 1 - \cos(A, B)
$$

Durante o treinamento, o modelo ajusta seus parâmetros para **maximizar a similaridade** entre embeddings de sentenças que são semanticamente próximas e **minimizar a similaridade** entre as que não são.

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
O notebook de treinamento pode ser acessado aqui: [`train_sbert_cosine.ipynb`](notebooks/train_sbert_cosine.ipynb)

---

## 📈 Avaliação
O notebook de avaliação do treinamento pode ser acessado aqui: [`evaluate_sbert.ipynb`](notebooks/evaluate_sbert.ipynb)

---

## 📊 Exemplo de resultado esperado
| sentence1                         | sentence2                               | similarity_score | predicted_similarity |
|-----------------------------------|------------------------------------------|------------------|----------------------|
| How do I learn ML?                | How to start learning machine learning?  | 4.8              | 4.6                  |
| What is the capital of Germany?   | How to become a software engineer?       | 1.3              | 1.2                  |

---

## 📜 Licença

Este projeto está licenciado sob os termos da **MIT License**.
Você é livre para usar, modificar e distribuir este código, desde que mantenha o aviso de copyright.
