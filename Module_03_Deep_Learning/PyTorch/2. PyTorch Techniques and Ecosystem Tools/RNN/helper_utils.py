import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd

import torchmetrics

from IPython.display import display,Markdown


def training_loop(model, train_loader, val_loader, loss_function, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_classes = len(train_loader.dataset.classes)
    avg_method = 'macro'

    # Initialize metrics
    val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    val_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average=avg_method).to(device)
    val_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average=avg_method).to(device)
    val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=avg_method).to(device)

    print(f"--- Training for {model.__class__.__name__} ---")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss_epoch = 0

        for batch in train_loader:
            optimizer.zero_grad()

            
            if model.__class__.__name__ == 'EmbeddingBagClassifier':
                text, offsets, labels = batch
                text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                outputs = model(text, offsets)
            else:
                text, labels = batch
                text, labels = text.to(device), labels.to(device)
                outputs = model(text)

            loss = loss_function(outputs, labels)
            train_loss_epoch += loss.item()

            loss.backward()
            optimizer.step()

        # FIXED: Divided by len(train_loader) OUTSIDE the batch loop
        train_loss_epoch /= len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_epoch = 0

        with torch.no_grad():
            for batch in val_loader:
                if model.__class__.__name__ == 'EmbeddingBagClassifier':
                    text, offsets, labels = batch
                    text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                    val_outputs = model(text, offsets)
                else:
                    text, labels = batch
                    text, labels = text.to(device), labels.to(device)
                    val_outputs = model(text)

                val_loss_epoch += loss_function(val_outputs, labels).item()
                
                # Update metrics
                val_accuracy.update(val_outputs, labels)
                val_precision.update(val_outputs, labels)
                val_recall.update(val_outputs, labels)
                val_f1.update(val_outputs, labels)

            val_loss_epoch /= len(val_loader)

            # Compute metrics as Python floats
            epoch_accuracy = val_accuracy.compute().item()
            epoch_precision = val_precision.compute().item()
            epoch_recall = val_recall.compute().item()
            epoch_f1 = val_f1.compute().item()

            # Reset metrics for next epoch
            val_accuracy.reset()
            val_precision.reset()
            val_recall.reset()
            val_f1.reset()

            if epoch % 1 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train Loss: {train_loss_epoch:.4f}, "
                    f"Val Loss: {val_loss_epoch:.4f}, "
                    f"Val Accuracy: {epoch_accuracy:.4f}"
                )

    print("--- Training completed ---")

    # FIXED: Removed the duplicate `.item()` calls
    final_results = {
        "val_accuracy": epoch_accuracy,
        "val_precision": epoch_precision,
        "val_recall": epoch_recall,
        "val_f1": epoch_f1
    }

    return model, final_results

def get_results(results_embag, results_mean, results_max, results_sum):
    results_df = pd.DataFrame(
        {
            "Model": ["EmbeddingBag", "Mean Pooling", "Max Pooling", "Sum Pooling"],
            "Accuracy": [
                results_embag["val_accuracy"],
                results_mean["val_accuracy"],
                results_max["val_accuracy"],
                results_sum["val_accuracy"]
            ],
            "Precision": [
                results_embag["val_precision"],
                results_mean["val_precision"],
                results_max["val_precision"],
                results_sum["val_precision"]
            ],
            "Recall": [
                results_embag["val_recall"],
                results_mean["val_recall"],
                results_max["val_recall"],
                results_sum["val_recall"]
            ],
            "F1 Score": [
                results_embag["val_f1"],
                results_mean["val_f1"],
                results_max["val_f1"],
                results_sum["val_f1"]
            ]
        }
    )
    
    results_df = results_df.set_index("Model")

    # This will now work without any errors!
    results_style = results_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}"
    })

    return results_style


def predict_catgory(model,text,vocab,preprocess_text,device):
    model.to(device)
    model.eval()

    processed_text = preprocess_text(text)
    indexed_text = vocab.encode(processed_text)


    with torch.no_grad():

        if model.__class__.__name__ == 'EmbeddingBagClassifier':
            
            text_tensor = torch.tensor(indexed_text, dtype=torch.long).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)

            outputs = model(text_tensor, offsets)
        else:
            text_tensor = torch.tensor(indexed_text, dtype=torch.long).unsqueeze(0).to(device)
            outputs = model(text_tensor)

    predicted_class = torch.argmax(outputs, dim=1).item()

    category = "Vegetable Recipe" if predicted_class == 1 else "Fruit Recipe"

    return category