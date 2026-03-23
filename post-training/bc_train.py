
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader
from bc_model import BCModel

class BCTrainer:
    def __init__(self, data_path="output/bc_data", model_path="output/bc_model.pt"):
        self.data_path = data_path
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, epochs=20, batch_size=256, lr=1e-3, patience=5):
        print(f"Loading data from {self.data_path}...")
        X_train = np.load(f"{self.data_path}/X_train.npy")
        y_train = np.load(f"{self.data_path}/y_train.npy")
        X_val = np.load(f"{self.data_path}/X_val.npy")
        y_val = np.load(f"{self.data_path}/y_val.npy")
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
        
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        model = BCModel(input_size, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        print("Training started...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4%}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.model_path)
                early_stop_counter = 0
                print(f"    [SAVED] New best validation loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break
                    
    def evaluate_test(self):
        print("Evaluating on test set...")
        X_test = np.load(f"{self.data_path}/X_test.npy")
        y_test = np.load(f"{self.data_path}/y_test.npy")
        
        input_size = X_test.shape[1]
        num_classes = len(np.unique(y_test))
        
        model = BCModel(input_size, num_classes).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = model(X_test_tensor)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, y_test_tensor)
            _, predicted = torch.max(output, 1)
            correct = (predicted == y_test_tensor).sum().item()
            acc = correct / y_test_tensor.size(0)
            
            _, top3 = torch.topk(output, 3, dim=1)
            y_expanded = y_test_tensor.view(-1, 1).expand_as(top3)
            correct_top3 = (top3 == y_expanded).any(dim=1).sum().item()
            acc_top3 = correct_top3 / y_test_tensor.size(0)
            
        print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.4%} | Top-3 Acc: {acc_top3:.4%}")
        
        results = {
            'error_rate': 1 - acc,
            'nll': loss.item(),
            'top3_error': 1 - acc_top3
        }
        return results

if __name__ == "__main__":
    trainer = BCTrainer()
    trainer.train()
    results = trainer.evaluate_test()
    
    # Write to comparison csv format (first 3 lines of output/bc_vs_ppo_comparison.csv)
    # We will just print them here for now.
    print(f"classification_error_rate,BC_policy,,,{results['error_rate']:.4f},")
    print(f"negative_log_likelihood,BC_policy,,,{results['nll']:.4f},")
    print(f"top3_error_rate,BC_policy,,,{results['top3_error']:.4f},")
