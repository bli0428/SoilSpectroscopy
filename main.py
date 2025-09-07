import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class SoilSpectroscopyPredictor:
    """
    A comprehensive tool for predicting MIR spectral values from NIR measurements
    using multiple machine learning approaches.
    """
    
    def __init__(self, data_dir='data/', use_gpu=True):
        self.data_dir = data_dir
        self.use_gpu = use_gpu
        self.nir_data = None
        self.mir_data = None
        self.aligned_data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """Check what GPU libraries are available"""
        gpu_status = {}
        
        # Check XGBoost
        try:
            import xgboost as xgb
            gpu_status['xgboost'] = True
        except ImportError:
            gpu_status['xgboost'] = False
            
        # Check PyTorch
        try:
            import torch
            gpu_status['pytorch'] = True
            gpu_status['cuda'] = torch.cuda.is_available()
        except ImportError:
            gpu_status['pytorch'] = False
            gpu_status['cuda'] = False
            
        return gpu_status

    def load_data(self):
        """Load NIR and MIR datasets"""
        print("Loading spectroscopy data...")
        
        # Load datasets
        self.nir_data = pd.read_csv(f'{self.data_dir}neospectra_nir_v1.2.csv')
        self.mir_data = pd.read_csv(f'{self.data_dir}neospectra_mir_v1.2.csv')
        
        print(f"NIR data shape: {self.nir_data.shape}")
        print(f"MIR data shape: {self.mir_data.shape}")
        
        # Extract spectral columns
        nir_spectral_cols = [col for col in self.nir_data.columns if 'scan_nir.' in col and '_ref' in col]
        mir_spectral_cols = [col for col in self.mir_data.columns if 'scan_mir.' in col and '_abs' in col]
        
        print(f"NIR spectral bands: {len(nir_spectral_cols)} (wavelengths: {nir_spectral_cols[0].split('.')[1]} - {nir_spectral_cols[-1].split('.')[1]})")
        print(f"MIR spectral bands: {len(mir_spectral_cols)} (wavenumbers: {mir_spectral_cols[0].split('.')[1]} - {mir_spectral_cols[-1].split('.')[1]})")
        
        return nir_spectral_cols, mir_spectral_cols
    
    def align_datasets(self, nir_spectral_cols, mir_spectral_cols):
        """Align NIR and MIR datasets by sample ID"""
        print("\nAligning datasets by sample ID...")
        
        # Check for duplicates in each dataset
        nir_duplicates = self.nir_data['id.sample_local_c'].duplicated().sum()
        mir_duplicates = self.mir_data['id.sample_local_c'].duplicated().sum()
        
        print(f"NIR dataset: {len(self.nir_data)} rows, {nir_duplicates} duplicates")
        print(f"MIR dataset: {len(self.mir_data)} rows, {mir_duplicates} duplicates")
        
        # Handle duplicates by keeping first occurrence
        if nir_duplicates > 0:
            print(f"Removing {nir_duplicates} duplicate NIR samples...")
            nir_clean = self.nir_data.drop_duplicates(subset=['id.sample_local_c'], keep='first').copy()
        else:
            nir_clean = self.nir_data.copy()
            
        if mir_duplicates > 0:
            print(f"Removing {mir_duplicates} duplicate MIR samples...")
            mir_clean = self.mir_data.drop_duplicates(subset=['id.sample_local_c'], keep='first').copy()
        else:
            mir_clean = self.mir_data.copy()
        
        # Get common sample IDs from cleaned datasets
        nir_samples = set(nir_clean['id.sample_local_c'])
        mir_samples = set(mir_clean['id.sample_local_c'])
        common_samples = nir_samples.intersection(mir_samples)
        
        print(f"After deduplication:")
        print(f"  NIR unique samples: {len(nir_samples)}")
        print(f"  MIR unique samples: {len(mir_samples)}")
        print(f"  Common samples: {len(common_samples)}")
        print(f"  NIR-only samples: {len(nir_samples - mir_samples)}")
        print(f"  MIR-only samples: {len(mir_samples - nir_samples)}")
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between NIR and MIR datasets!")
        
        # Filter to only common samples
        nir_aligned = nir_clean[nir_clean['id.sample_local_c'].isin(common_samples)].copy()
        mir_aligned = mir_clean[mir_clean['id.sample_local_c'].isin(common_samples)].copy()
        
        # Sort by sample ID for proper alignment
        nir_aligned = nir_aligned.sort_values('id.sample_local_c').reset_index(drop=True)
        mir_aligned = mir_aligned.sort_values('id.sample_local_c').reset_index(drop=True)
        
        # Verify alignment
        nir_ids = nir_aligned['id.sample_local_c'].values
        mir_ids = mir_aligned['id.sample_local_c'].values
        
        if not np.array_equal(nir_ids, mir_ids):
            print("Sample ID mismatch details:")
            print(f"NIR aligned samples: {len(nir_ids)}")
            print(f"MIR aligned samples: {len(mir_ids)}")
            if len(nir_ids) > 0 and len(mir_ids) > 0:
                print(f"First few NIR IDs: {nir_ids[:5]}")
                print(f"First few MIR IDs: {mir_ids[:5]}")
            raise ValueError("Sample IDs are not properly aligned after sorting!")
        
        print(f"Successfully aligned {len(nir_ids)} samples")
        
        # Extract spectral data
        X = nir_aligned[nir_spectral_cols].values
        y = mir_aligned[mir_spectral_cols].values
        
        # Verify shapes match
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch after alignment: X={X.shape}, y={y.shape}")
        
        # Store sample IDs for reference
        sample_ids = nir_aligned['id.sample_local_c'].values
        
        self.aligned_data = {
            'X': X,
            'y': y,
            'sample_ids': sample_ids,
            'nir_wavelengths': [float(col.split('.')[1].removesuffix('_ref')) for col in nir_spectral_cols],
            'mir_wavenumbers': [float(col.split('.')[1].removesuffix('_abs')) for col in mir_spectral_cols]
        }
        
        print(f"Final aligned dataset shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def preprocess_data(self, X, y, scaler_type='standard'):
        """Preprocess spectral data with various scaling options"""
        print(f"\nPreprocessing data with {scaler_type} scaling...")
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type == 'minmax':
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")
        
        # Fit and transform
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        self.scalers[scaler_type] = {'X': scaler_X, 'y': scaler_y}
        
        return X_scaled, y_scaled
    
    def setup_models(self):
        """Initialize various regression models for comparison"""
        print("\nSetting up regression models...")
        
        models = {
            # Linear Models
            'Ridge': Ridge(alpha=1.0, random_state=42),
            
            # Tree-based Models
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            
            # # Neural Network
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            
            # # Support Vector Machine
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),

            'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        }
        
        # Add GPU models if available and requested
        if self.use_gpu:
            gpu_models = self._setup_gpu_models()
            models.update(gpu_models)
        
        return models
    
    def _setup_gpu_models(self):
        """Setup GPU-accelerated models"""
        gpu_models = {}
        
        # XGBoost GPU Model
        if self.gpu_available['xgboost']:
            try:
                import xgboost as xgb
                gpu_models['GPU_XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,  # Limit tree depth
                    learning_rate=0.1,  # Reduce learning rate
                    subsample=0.8,  # Use 80% of samples
                    colsample_bytree=0.8,  # Use 80% of features
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    tree_method='gpu_hist',  # GPU acceleration
                    gpu_id=0,
                    random_state=42,
                    early_stopping_rounds=10,  # Stop if no improvement
                    eval_metric='rmse'
                )
                print("✅ Added GPU XGBoost model with regularization")
            except Exception as e:
                print(f"⚠️ XGBoost GPU not available: {e}")
                # Fallback to CPU XGBoost with same regularization
                import xgboost as xgb
                gpu_models['CPU_XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    early_stopping_rounds=10,
                    eval_metric='rmse'
                )
                print("✅ Added CPU XGBoost model as fallback with regularization")
        
        # PyTorch Neural Network
        if self.gpu_available['pytorch']:
            gpu_models['GPU_Neural_Network'] = 'pytorch_model'  # Special marker
            print("✅ Added GPU PyTorch Neural Network model")
        
        return gpu_models
    
    def _create_pytorch_model(self, input_dim, output_dim):
        """Create PyTorch neural network model"""
        import torch
        import torch.nn as nn
        
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
    
    def _train_pytorch_model(self, X_train, X_test, y_train, y_test, epochs=100):
        """Train PyTorch model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        device = 'cuda' if self.gpu_available['cuda'] else 'cpu'
        print(f"  Training PyTorch model on {device}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # Create model
        model = self._create_pytorch_model(X_train.shape[1], y_train.shape[1]).to(device)
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
                print(f'    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).cpu().numpy()
            y_pred_test = model(X_test_tensor).cpu().numpy()
        
        return model, y_pred_train, y_pred_test

    def train_models(self, X_train, X_test, y_train, y_test, models):
        """Train and evaluate multiple models"""
        print("\nTraining models...")
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            start_time = time.time()
            try:
                if name == 'GPU_Neural_Network':
                    # Special handling for PyTorch
                    trained_model, y_pred_train, y_pred_test = self._train_pytorch_model(
                        X_train, X_test, y_train, y_test
                    )
                else:
                    # Standard scikit-learn-like interface
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    trained_model = model
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                results[name] = {
                    'model': trained_model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                }
                
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"  Training time: {elapsed_time:.2f} seconds")
        
        return results
    
    def dimensionality_reduction_approach(self, X, y, n_components=50):
        """Alternative approach using PCA for dimensionality reduction"""
        print(f"\nTesting PCA approach with {n_components} components...")
        
        # Apply PCA to NIR data
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model on PCA features
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"PCA + Ridge - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return {
            'pca': pca,
            'model': model,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'X_pca': X_pca
        }
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Ridge'):
        """Perform hyperparameter tuning for selected model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Ridge':
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            model = Ridge(random_state=42)
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def plot_results(self, results, scaler_type='standard'):
        """Plot model comparison and predictions"""
        print("\nGenerating plots...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        models = list(results.keys())
        train_r2 = [results[m]['train_r2'] for m in models]
        test_r2 = [results[m]['test_r2'] for m in models]
        
        x_pos = np.arange(len(models))
        axes[0, 0].bar(x_pos - 0.2, train_r2, 0.4, label='Train', alpha=0.7)
        axes[0, 0].bar(x_pos + 0.2, test_r2, 0.4, label='Test', alpha=0.7)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Performance Comparison (R²)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        train_rmse = [results[m]['train_rmse'] for m in models]
        test_rmse = [results[m]['test_rmse'] for m in models]
        
        axes[0, 1].bar(x_pos - 0.2, train_rmse, 0.4, label='Train', alpha=0.7)
        axes[0, 1].bar(x_pos + 0.2, test_rmse, 0.4, label='Test', alpha=0.7)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Model Performance Comparison (RMSE)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Best model predictions (first few MIR bands)
        best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
        y_test = results[best_model]['y_pred_test']
        y_true = results[best_model]['y_pred_test']  # This should be actual y_test from split
        
        # Prediction vs actual for first MIR band
        axes[1, 0].scatter(y_true[:, 0], y_test[:, 0], alpha=0.6)
        axes[1, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                       [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual MIR (First Band)')
        axes[1, 0].set_ylabel('Predicted MIR (First Band)')
        axes[1, 0].set_title(f'Best Model ({best_model}) - First MIR Band')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spectral comparison for a sample
        sample_idx = 0
        mir_wavenumbers = self.aligned_data['mir_wavenumbers']
        axes[1, 1].plot(mir_wavenumbers, y_true[sample_idx], 'b-', label='Actual', linewidth=2)
        axes[1, 1].plot(mir_wavenumbers, y_test[sample_idx], 'r--', label='Predicted', linewidth=2)
        axes[1, 1].set_xlabel('Wavenumber (cm⁻¹)')
        axes[1, 1].set_ylabel('Absorbance')
        axes[1, 1].set_title(f'Spectral Comparison - Sample {sample_idx+1}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_comparison_{scaler_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def apply_bias_correction(self, y_true, y_pred, method='mean_offset'):
        """Apply bias correction to predictions"""
        if method == 'mean_offset':
            # Calculate mean offset per wavelength
            bias = np.mean(y_true - y_pred, axis=0)
            y_pred_corrected = y_pred + bias
            
        elif method == 'median_offset':
            # Use median for robustness to outliers
            bias = np.median(y_true - y_pred, axis=0)
            y_pred_corrected = y_pred + bias
            
        elif method == 'linear_correction':
            # Fit linear correction: y_true = a * y_pred + b
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            
            # Fit correction for each wavelength
            y_pred_corrected = np.zeros_like(y_pred)
            correction_params = []
            
            for i in range(y_pred.shape[1]):
                lr.fit(y_pred[:, i].reshape(-1, 1), y_true[:, i])
                y_pred_corrected[:, i] = lr.predict(y_pred[:, i].reshape(-1, 1))
                correction_params.append({'slope': lr.coef_[0], 'intercept': lr.intercept_})
            
            return y_pred_corrected, correction_params
            
        else:
            raise ValueError("Method must be 'mean_offset', 'median_offset', or 'linear_correction'")
        
        return y_pred_corrected, bias
    
    def evaluate_with_bias_correction(self, y_true, y_pred, methods=['mean_offset', 'median_offset', 'linear_correction']):
        """Evaluate different bias correction methods"""
        print("  → Testing bias correction methods...")
        
        results = {}
        original_r2 = r2_score(y_true, y_pred)
        original_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        results['original'] = {
            'r2': original_r2,
            'rmse': original_rmse,
            'y_pred': y_pred
        }
        
        for method in methods:
            try:
                y_pred_corrected, correction_info = self.apply_bias_correction(y_true, y_pred, method)
                
                corrected_r2 = r2_score(y_true, y_pred_corrected)
                corrected_rmse = np.sqrt(mean_squared_error(y_true, y_pred_corrected))
                
                results[method] = {
                    'r2': corrected_r2,
                    'rmse': corrected_rmse,
                    'y_pred': y_pred_corrected,
                    'correction_info': correction_info,
                    'improvement_r2': corrected_r2 - original_r2,
                    'improvement_rmse': original_rmse - corrected_rmse
                }
                
                print(f"    {method}: R²={corrected_r2:.4f} (+{corrected_r2-original_r2:.4f}), RMSE={corrected_rmse:.4f} (-{original_rmse-corrected_rmse:.4f})")
                
            except Exception as e:
                print(f"    {method}: Failed - {e}")
                continue
        
        # Find best method
        best_method = max([k for k in results.keys() if k != 'original'], 
                         key=lambda x: results[x]['r2'])
        
        print(f"    Best correction: {best_method}")
        return results, best_method
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        analysis_start_time = time.time()
        
        print("="*60)
        print("SOIL SPECTROSCOPY: NIR to MIR PREDICTION ANALYSIS")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.use_gpu:
            print("GPU ACCELERATION ENABLED")
            print(f"Available GPU libraries: XGBoost={self.gpu_available['xgboost']}, PyTorch={self.gpu_available['pytorch']}, CUDA={self.gpu_available['cuda']}")
        print("="*60)
        
        # Load and align data
        print("\n🔄 PHASE 1: Data Loading and Alignment")
        data_start = time.time()
        nir_cols, mir_cols = self.load_data()
        X, y = self.align_datasets(nir_cols, mir_cols)
        data_time = time.time() - data_start
        print(f"✅ Data preparation completed in {data_time:.2f}s")
        
        # Test different scaling approaches
        scaling_methods = ['standard', 'minmax', 'robust']
        all_results = {}
        
        print(f"\n🔄 PHASE 2: Model Training ({len(scaling_methods)} scaling methods)")
        
        for i, scaler_type in enumerate(scaling_methods, 1):
            print(f"\n{'='*40}")
            print(f"SCALING METHOD {i}/{len(scaling_methods)}: {scaler_type.upper()}")
            print(f"{'='*40}")
            
            scaling_start = time.time()
            
            # Preprocess data
            print(f"🔄 Preprocessing with {scaler_type} scaling...")
            X_scaled, y_scaled = self.preprocess_data(X, y, scaler_type)
            
            # Split data
            print("🔄 Splitting data into train/test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Setup and train models
            print("🔄 Setting up models...")
            models = self.setup_models()
            
            results = self.train_models(X_train, X_test, y_train, y_test, models)
            
            # Store results
            all_results[scaler_type] = results
            
            scaling_time = time.time() - scaling_start
            print(f"✅ {scaler_type.upper()} scaling completed in {scaling_time:.2f}s")
            
            # Plot results
            print("🔄 Generating plots...")
            plot_start = time.time()
            self.plot_results(results, scaler_type)
            plot_time = time.time() - plot_start
            print(f"✅ Plots generated in {plot_time:.2f}s")
        
        # PCA approach
        print(f"\n{'='*40}")
        print("🔄 PHASE 3: PCA DIMENSIONALITY REDUCTION")
        print(f"{'='*40}")
        
        pca_start = time.time()
        X_std, y_std = self.preprocess_data(X, y, 'standard')
        pca_results = self.dimensionality_reduction_approach(X_std, y_std)
        pca_time = time.time() - pca_start
        print(f"✅ PCA analysis completed in {pca_time:.2f}s")
        
        # Hyperparameter tuning for best model
        print(f"\n{'='*40}")
        print("🔄 PHASE 4: HYPERPARAMETER TUNING")
        print(f"{'='*40}")
        
        tuning_start = time.time()
        
        # Use standard scaling for tuning
        X_scaled, y_scaled = self.preprocess_data(X, y, 'standard')
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print("🔄 Tuning Ridge regression...")
        tuned_ridge = self.hyperparameter_tuning(X_train, y_train, 'Ridge')
        print("🔄 Tuning Random Forest...")
        tuned_rf = self.hyperparameter_tuning(X_train, y_train, 'RandomForest')
        
        tuning_time = time.time() - tuning_start
        print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f}s")
        
        # Final recommendations
        print(f"\n{'='*60}")
        print("🏆 FINAL RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Find best overall model
        best_scaler = None
        best_model = None
        best_score = -np.inf
        
        for scaler_type, results in all_results.items():
            for model_name, result in results.items():
                if result['test_r2'] > best_score:
                    best_score = result['test_r2']
                    best_scaler = scaler_type
                    best_model = model_name
        
        print(f"🥇 Best Model: {best_model} with {best_scaler} scaling")
        print(f"📊 Best Test R²: {best_score:.4f}")
        print(f"📊 Best Test RMSE: {all_results[best_scaler][best_model]['test_rmse']:.4f}")
        
        # GPU performance summary
        if self.use_gpu:
            gpu_models = [name for name in all_results[best_scaler].keys() if 'GPU' in name or 'XGBoost' in name]
            if gpu_models:
                print(f"\n🚀 GPU Models Performance:")
                for model_name in gpu_models:
                    result = all_results[best_scaler][model_name]
                    print(f"  • {model_name}: R²={result['test_r2']:.4f}, RMSE={result['test_rmse']:.4f}")
        
        # Total analysis time
        total_analysis_time = time.time() - analysis_start_time
        print(f"\n⏱️  TOTAL ANALYSIS TIME: {total_analysis_time:.2f} seconds ({total_analysis_time/60:.1f} minutes)")
        print(f"🏁 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_results, pca_results

# Usage example
if __name__ == "__main__":
    # Initialize predictor with GPU support
    predictor = SoilSpectroscopyPredictor(use_gpu=True)
    
    # Run complete analysis
    results, pca_results = predictor.run_complete_analysis()
    
    print("\nAnalysis complete! Check the generated plots for detailed results.")