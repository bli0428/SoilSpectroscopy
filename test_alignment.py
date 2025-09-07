#!/usr/bin/env python3
"""
Test script to verify the sample alignment fix works correctly
"""

from main import SoilSpectroscopyPredictor

def test_alignment():
    """Test the alignment functionality"""
    print("Testing sample alignment fix...")
    
    try:
        # Initialize predictor
        predictor = SoilSpectroscopyPredictor()
        
        # Load data
        nir_cols, mir_cols = predictor.load_data()
        
        # Test alignment
        X, y = predictor.align_datasets(nir_cols, mir_cols)
        
        print(f"\n✅ Alignment successful!")
        print(f"Final shapes: X={X.shape}, y={y.shape}")
        print(f"NIR wavelengths: {len(predictor.aligned_data['nir_wavelengths'])}")
        print(f"MIR wavenumbers: {len(predictor.aligned_data['mir_wavenumbers'])}")
        
        # Test preprocessing
        print("\nTesting preprocessing...")
        X_scaled, y_scaled = predictor.preprocess_data(X, y, 'standard')
        print(f"Scaled shapes: X_scaled={X_scaled.shape}, y_scaled={y_scaled.shape}")
        
        print("\n✅ All tests passed! Ready to run full analysis.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_alignment()
    if success:
        print("\nYou can now run the full analysis with:")
        print("python main.py")
    else:
        print("\nPlease check the error above and fix any issues.")
