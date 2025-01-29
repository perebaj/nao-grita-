import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def prepare_data(data):
    """
    Prepara os dados para treinamento, normalizando as notas para classes
    """
    # Converte notas de 0-10 para 0-5
    normalized_ratings = [round(rate / 2) for rate in data["rate"]]

    # Combina texto com descrição do trabalho e conselho
    texts = [f"{row['work_description']} {row['advice_description']}" for _, row in data.iterrows()]

    return {"text": texts, "label": normalized_ratings}


def compute_metrics(pred):
    """
    Calcula métricas de avaliação do modelo
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_model(data_df, output_dir="./data"):
    """
    Realiza o fine-tuning do modelo
    """
    # Prepara os dados
    processed_data = prepare_data(data_df)
    dataset = Dataset.from_dict(processed_data)

    # Carrega o tokenizer e o modelo
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt", num_labels=6, ignore_mismatched_sizes=True  # 0-5 ratings
    )

    # Reinicializa a camada de classificação com os pesos corretos
    model.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(model.config.hidden_size, 6))  # 6 classes (0-5)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # Tokeniza o dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Cria o trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Idealmente deveria ser um conjunto separado
        compute_metrics=compute_metrics,
    )

    # Treina o modelo
    trainer.train()

    # Salva o modelo e tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def predict_rating(text, model, tokenizer):
    """
    Faz predição de nota para um novo texto
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)

    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class


# Exemplo de uso
if __name__ == "__main__":
    # Seus dados de exemplo
    data = {
        "rate": [2.0, 10.0],
        "work_description": [
            "Terrível... Não trabalha com o que foi combinado!",
            "Ela é incrível, a campanha foi ótima!",
        ],
        "advice_description": ["Não espere datas cumpridas...", "Podem contratarr hahahah"],
    }

    df = pd.DataFrame(data)

    # Treina o modelo
    model, tokenizer = train_model(df)

    # Exemplo de predição
    new_text = "Profissional excelente, entregou tudo no prazo"
    predicted_rating = predict_rating(new_text, model, tokenizer)
    print(f"Nota prevista: {predicted_rating}/5")
