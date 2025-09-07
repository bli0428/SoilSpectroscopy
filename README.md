# Soil Spectroscopy: NIR to MIR Prediction Tool

This tool predicts high-quality MIR (Mid-Infrared) spectral values from lower-quality NIR (Near-Infrared) measurements for soil samples using multiple machine learning approaches.

## Overview

The tool addresses the common challenge in soil spectroscopy where high-quality MIR measurements are expensive and time-consuming, while NIR measurements are faster and more cost-effective but provide lower spectral resolution. By training models on paired NIR-MIR data, we can predict MIR spectra from new NIR measurements.

## Data Structure

- **NIR Data**: `neospectra_nir_v1.2.csv` - Near-infrared reflectance measurements (1350-2550 nm)
- **MIR Data**: `neospectra_mir_v1.2.csv` - Mid-infrared absorbance measurements (600-4000 cm⁻¹)
- Both datasets contain measurements for the same soil samples with matching `id.sample_local_c`

## Modeling Approaches

### 1. Linear Regression Models

#### Ridge Regression
- **Best for**: High-dimensional spectral data with multicollinearity
- **Advantages**: Handles overfitting well, stable predictions
- **Use case**: When you need interpretable, robust predictions

#### Lasso Regression
- **Best for**: Feature selection and sparse solutions
- **Advantages**: Automatic feature selection, simpler models
- **Use case**: When you want to identify most important NIR wavelengths

#### Elastic Net
- **Best for**: Combining Ridge and Lasso benefits
- **Advantages**: Balanced regularization, handles grouped features
- **Use case**: When you have correlated spectral bands

### 2. Tree-Based Models

#### Random Forest
- **Best for**: Non-linear relationships, robust predictions
- **Advantages**: Handles non-linearity, provides feature importance
- **Use case**: When spectral relationships are complex and non-linear

#### Gradient Boosting
- **Best for**: High accuracy, sequential learning
- **Advantages**: Often highest performance, handles complex patterns
- **Use case**: When maximum prediction accuracy is priority

### 3. Neural Networks

#### Multi-Layer Perceptron (MLP)
- **Best for**: Complex non-linear mappings
- **Advantages**: Can learn complex spectral transformations
- **Use case**: When you have sufficient data and complex spectral relationships

### 4. Support Vector Regression (SVR)
- **Best for**: High-dimensional data with non-linear patterns
- **Advantages**: Effective in high dimensions, memory efficient
- **Use case**: When you need robust performance with limited data

### 5. Dimensionality Reduction Approach

#### PCA + Ridge Regression
- **Best for**: Reducing computational complexity
- **Advantages**: Faster training, noise reduction
- **Use case**: When you have limited computational resources or noisy data

## Data Preprocessing Options

### Scaling Methods

1. **Standard Scaling** (Z-score normalization)
   - Centers data around mean=0, std=1
   - Best for: Most algorithms, especially neural networks

2. **Min-Max Scaling**
   - Scales to [0,1] range
   - Best for: When you need bounded outputs

3. **Robust Scaling**
   - Uses median and IQR, less sensitive to outliers
   - Best for: Data with outliers or non-normal distributions

## Usage

### Basic Usage

```python
from main import SoilSpectroscopyPredictor

# Initialize predictor
predictor = SoilSpectroscopyPredictor(data_dir='data/')

# Run complete analysis
results, pca_results = predictor.run_complete_analysis()
```

### Custom Analysis

```python
# Load and align data
nir_cols, mir_cols = predictor.load_data()
X, y = predictor.align_datasets(nir_cols, mir_cols)

# Preprocess with specific scaling
X_scaled, y_scaled = predictor.preprocess_data(X, y, scaler_type='standard')

# Train specific models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

models = predictor.setup_models()
results = predictor.train_models(X_train, X_test, y_train, y_test, models)
```

### Hyperparameter Tuning

```python
# Tune specific model
best_ridge = predictor.hyperparameter_tuning(X_train, y_train, 'Ridge')
best_rf = predictor.hyperparameter_tuning(X_train, y_train, 'RandomForest')
```

## Model Selection Guidelines

### For High Accuracy
1. **Gradient Boosting** - Often provides best performance
2. **Random Forest** - Good balance of accuracy and interpretability
3. **Tuned Neural Network** - For complex spectral relationships

### For Speed/Efficiency
1. **Ridge Regression** - Fast training and prediction
2. **PCA + Ridge** - Reduced dimensionality for speed
3. **Linear models** - Minimal computational requirements

### For Interpretability
1. **Ridge Regression** - Linear relationships, coefficient interpretation
2. **Lasso Regression** - Feature selection insights
3. **Random Forest** - Feature importance rankings

### For Robustness
1. **Random Forest** - Handles outliers and noise well
2. **Robust Scaling + Ridge** - Less sensitive to outliers
3. **Cross-validated models** - More stable performance estimates

## Evaluation Metrics

- **R² Score**: Proportion of variance explained (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## Output Files

The analysis generates several visualization files:
- `model_comparison_standard.png` - Results with standard scaling
- `model_comparison_minmax.png` - Results with min-max scaling  
- `model_comparison_robust.png` - Results with robust scaling

## Installation

```bash
pip install -r requirements.txt
```

## Expected Performance

Typical performance ranges for soil spectroscopy NIR→MIR prediction:
- **Good models**: R² > 0.85, RMSE < 0.1
- **Excellent models**: R² > 0.90, RMSE < 0.05
- **Production-ready**: R² > 0.92, RMSE < 0.03

## Recommendations

### For Production Use
1. Use **Gradient Boosting** or **Random Forest** for best accuracy
2. Apply **Standard Scaling** for preprocessing
3. Perform **hyperparameter tuning** on selected model
4. Validate on independent test set
5. Monitor model performance over time

### For Research/Exploration
1. Compare all models to understand data characteristics
2. Analyze feature importance from tree-based models
3. Use PCA to understand spectral relationships
4. Test different preprocessing approaches

### For Real-time Applications
1. Use **Ridge Regression** for speed
2. Consider **PCA preprocessing** to reduce dimensions
3. Pre-train and save models for inference
4. Implement batch prediction for efficiency

## Troubleshooting

### Low Performance (R² < 0.8)
- Check data quality and alignment
- Try different scaling methods
- Increase model complexity (Random Forest, Neural Networks)
- Consider feature engineering

### Overfitting (Large train-test gap)
- Increase regularization (higher alpha for Ridge/Lasso)
- Reduce model complexity
- Use more training data
- Apply cross-validation

### Slow Training
- Use PCA for dimensionality reduction
- Reduce number of estimators for tree models
- Use simpler models (Ridge instead of Random Forest)
- Consider data sampling for large datasets

## Future Enhancements

- Deep learning approaches (CNN, LSTM for spectral sequences)
- Transfer learning from pre-trained spectral models
- Ensemble methods combining multiple approaches
- Real-time inference API
- Uncertainty quantification
- Domain adaptation for different soil types
