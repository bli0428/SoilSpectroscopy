#!/usr/bin/env python3
"""
GPU-accelerated models for soil spectroscopy prediction
Requires: cuml, xgboost, pytorch, cupy (optional)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Check for GPU libraries availability
GPU_AVAILABLE = {}

try:
    import cuml
    from cuml.linear_model import Ridge as CuRidge, Lasso as CuLasso, ElasticNet as CuElasticNet
    from cuml.ensemble import RandomForestRegressor as CuRandomForest
    from cuml.svm import SVR as CuSVR
    from cuml.decomposition import PCA as CuPCA
    GPU_AVAILABLE['cuml'] = True
    print("✅ CuML (RAPIDS) available for GPU acceleration")
except ImportError:
    GPU_AVAILABLE['cuml'] = False
    print("❌ CuML not available. Install with: conda install -c rapidsai cuml")

try:
    import xgboost as xgb
    GPU_AVAILABLE['xgboost'] = True
    print("✅ XGBoost available (check for GPU support)")
except ImportError:
    GPU_AVAILABLE['xgboost'] = False
    print("❌ XGBoost not available. Install with: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    GPU_AVAILABLE['pytorch'] = True
    GPU_AVAILABLE['cuda'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        print(f"✅ PyTorch with CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ PyTorch available but no CUDA GPU detected")
except ImportError:
    GPU_AVAILABLE['pytorch'] = False
    GPU_AVAILABLE['cuda'] = False
    print("❌ PyTorch not available. Install with: pip install torch")

try:
    import cupy as cp
    GPU_AVAILABLE['cupy'] = True
    print("✅ CuPy available for GPU-accelerated NumPy operations")
except ImportError:
    GPU_AVAILABLE['cupy'] = False
    print("❌ CuPy not available. Install with: pip install cupy-cuda11x")

class GPUSpectroscopyPredictor:
    """GPU-accelerated version of soil spectroscopy predictor"""
    
    def __init__(self):
        self.models = {}
        self.device = 'cuda' if GPU_AVAILABLE.get('cuda', False) else 'cpu'
        print(f"Using device: {self.device}")
    
    def setup_gpu_models(self):
        """Setup GPU-accelerated models"""
        models = {}
        
        # CuML Models (if available)
        if GPU_AVAILABLE['cuml']:
            models.update({
                'GPU_Ridge': CuRidge(alpha=1.0),
                'GPU_Lasso': CuLasso(alpha=0.1, max_iter=2000),
                'GPU_ElasticNet': CuElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
                'GPU_RandomForest': CuRandomForest(n_estimators=100, random_state=42),
                'GPU_SVR': CuSVR(kernel='rbf', C=1.0, gamma='scale')
            })
        
        # XGBoost GPU Model
        if GPU_AVAILABLE['xgboost']:
            try:
                models['GPU_XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    tree_method='gpu_hist',  # GPU acceleration
                    gpu_id=0,
                    random_state=42
                )
            except Exception as e:
                print(f"⚠️ XGBoost GPU not available: {e}")
                models['CPU_XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42
                )
        
        # PyTorch Neural Network
        if GPU_AVAILABLE['pytorch']:
            models['GPU_Neural_Network'] = self._create_pytorch_model
        
        return models
    
    def _create_pytorch_model(self, input_dim, output_dim):
        """Create PyTorch neural network model"""
        class SpectroscopyNet(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(SpectroscopyNet, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return SpectroscopyNet(input_dim, output_dim)
    
    def train_pytorch_model(self, X_train, X_test, y_train, y_test, epochs=100, batch_size=64):
        """Train PyTorch model on GPU"""
        if not GPU_AVAILABLE['pytorch']:
            return None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Create model
        model = self._create_pytorch_model(X_train.shape[1], y_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).cpu().numpy()
            y_pred_test = model(X_test_tensor).cpu().numpy()
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
    
    def train_gpu_models(self, X_train, X_test, y_train, y_test):
        """Train all available GPU models"""
        print("\nTraining GPU-accelerated models...")
        
        models = self.setup_gpu_models()
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                if name == 'GPU_Neural_Network':
                    # Special handling for PyTorch
                    result = self.train_pytorch_model(X_train, X_test, y_train, y_test)
                    if result:
                        y_pred_train = result['y_pred_train']
                        y_pred_test = result['y_pred_test']
                        trained_model = result['model']
                    else:
                        continue
                else:
                    # Standard scikit-learn-like interface
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    trained_model = model
                
                # Convert CuML arrays to numpy if needed
                if hasattr(y_pred_train, 'to_numpy'):
                    y_pred_train = y_pred_train.to_numpy()
                if hasattr(y_pred_test, 'to_numpy'):
                    y_pred_test = y_pred_test.to_numpy()
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                results[name] = {
                    'model': trained_model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                }
                
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
        
        return results
    
    def gpu_pca_approach(self, X, y, n_components=50):
        """GPU-accelerated PCA approach"""
        if not GPU_AVAILABLE['cuml']:
            print("CuML not available for GPU PCA")
            return None
        
        print(f"\nTesting GPU PCA approach with {n_components} components...")
        
        # Apply GPU PCA
        pca = CuPCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Convert to numpy if needed
        if hasattr(X_pca, 'to_numpy'):
            X_pca = X_pca.to_numpy()
        
        explained_var = pca.explained_variance_ratio_.sum()
        if hasattr(explained_var, 'to_numpy'):
            explained_var = float(explained_var.to_numpy())
        
        print(f"GPU PCA explained variance ratio: {explained_var:.4f}")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )
        
        model = CuRidge(alpha=1.0)
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        if hasattr(y_pred_test, 'to_numpy'):
            y_pred_test = y_pred_test.to_numpy()
        
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"GPU PCA + Ridge - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return {
            'pca': pca,
            'model': model,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }

def benchmark_gpu_vs_cpu(X, y, model_name='RandomForest'):
    """Benchmark GPU vs CPU performance"""
    import time
    from sklearn.ensemble import RandomForestRegressor as CPURandomForest
    
    print(f"\nBenchmarking {model_name}: GPU vs CPU")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CPU Model
    print("Training CPU model...")
    start_time = time.time()
    cpu_model = CPURandomForest(n_estimators=100, random_state=42, n_jobs=-1)
    cpu_model.fit(X_train, y_train)
    cpu_pred = cpu_model.predict(X_test)
    cpu_time = time.time() - start_time
    cpu_r2 = r2_score(y_test, cpu_pred)
    
    # GPU Model (if available)
    if GPU_AVAILABLE['cuml']:
        print("Training GPU model...")
        start_time = time.time()
        gpu_model = CuRandomForest(n_estimators=100, random_state=42)
        gpu_model.fit(X_train, y_train)
        gpu_pred = gpu_model.predict(X_test)
        if hasattr(gpu_pred, 'to_numpy'):
            gpu_pred = gpu_pred.to_numpy()
        gpu_time = time.time() - start_time
        gpu_r2 = r2_score(y_test, gpu_pred)
        
        # Results
        speedup = cpu_time / gpu_time
        print(f"\nResults:")
        print(f"CPU Time: {cpu_time:.2f}s, R²: {cpu_r2:.4f}")
        print(f"GPU Time: {gpu_time:.2f}s, R²: {gpu_r2:.4f}")
        print(f"Speedup: {speedup:.2f}x")
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cpu_r2': cpu_r2,
            'gpu_r2': gpu_r2
        }
    else:
        print("GPU model not available for comparison")
        return {'cpu_time': cpu_time, 'cpu_r2': cpu_r2}

# Usage example
if __name__ == "__main__":
    print("GPU Spectroscopy Predictor")
    print("="*40)
    
    # Check what's available
    available_models = []
    if GPU_AVAILABLE['cuml']:
        available_models.extend(['Ridge', 'ElasticNet', 'RandomForest', 'SVR'])
        # available_models.append('Lasso')
    if GPU_AVAILABLE['xgboost']:
        available_models.append('XGBoost')
    if GPU_AVAILABLE['pytorch']:
        available_models.append('Neural Network')
    
    print(f"Available GPU models: {available_models}")
    
    if not available_models:
        print("\n❌ No GPU libraries available. Install:")
        print("  - CuML: conda install -c rapidsai cuml")
        print("  - XGBoost: pip install xgboost")
        print("  - PyTorch: pip install torch")
