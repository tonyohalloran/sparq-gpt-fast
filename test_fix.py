#!/usr/bin/env python3
"""
Test script to verify that the SparQ fixes work correctly.
"""

import torch
from model import Transformer, ModelArgs, DenseAttentionFunction
from sparq import SparQArgs, RKForCompressionRatio

def test_dense_attention():
    """Test that dense attention works correctly."""
    print("Testing dense attention...")
    
    config = ModelArgs(
        attention="dense",
        sparq=SparQArgs(rk=RKForCompressionRatio(ratio=0))  # This should be ignored
    )
    
    model = Transformer(config)
    print("✓ Dense attention model created successfully")
    
    # Test that it uses DenseAttentionFunction
    attention_func = model.layers[0].attention.attention_function
    assert isinstance(attention_func, DenseAttentionFunction), f"Expected DenseAttentionFunction, got {type(attention_func)}"
    print("✓ Model uses DenseAttentionFunction")
    
    return True

def test_sparq_attention():
    """Test that SparQ attention works correctly with valid compression ratio."""
    print("Testing SparQ attention...")
    
    config = ModelArgs(
        attention="sparq",
        sparq=SparQArgs(rk=RKForCompressionRatio(ratio=8))
    )
    
    model = Transformer(config)
    print("✓ SparQ attention model created successfully")
    
    # Test that it uses SparQAttentionFunction
    from sparq import SparQAttentionFunction
    attention_func = model.layers[0].attention.attention_function
    assert isinstance(attention_func, SparQAttentionFunction), f"Expected SparQAttentionFunction, got {type(attention_func)}"
    print("✓ Model uses SparQAttentionFunction")
    
    return True

def test_invalid_sparq_ratio():
    """Test that invalid SparQ compression ratio falls back to dense attention."""
    print("Testing invalid SparQ compression ratio...")
    
    config = ModelArgs(
        attention="sparq",
        sparq=SparQArgs(rk=RKForCompressionRatio(ratio=0))  # Invalid ratio
    )
    
    model = Transformer(config)
    print("✓ Model created successfully (should fall back to dense)")
    
    # Test that it falls back to DenseAttentionFunction
    attention_func = model.layers[0].attention.attention_function
    assert isinstance(attention_func, DenseAttentionFunction), f"Expected fallback to DenseAttentionFunction, got {type(attention_func)}"
    print("✓ Model correctly fell back to DenseAttentionFunction")
    
    return True

def test_compression_ratio_validation():
    """Test compression ratio validation."""
    print("Testing compression ratio validation...")
    
    from sparq import get_r_k_for_compression_ratio
    
    # Test valid ratio
    try:
        r, k = get_r_k_for_compression_ratio(8, 1000, 128)
        print(f"✓ Valid ratio 8: r={r}, k={k}")
    except Exception as e:
        print(f"✗ Valid ratio failed: {e}")
        return False
    
    # Test invalid ratio
    try:
        get_r_k_for_compression_ratio(0, 1000, 128)
        print("✗ Invalid ratio 0 should have failed")
        return False
    except ValueError as e:
        print(f"✓ Invalid ratio 0 correctly failed: {e}")
    
    # Test negative ratio
    try:
        get_r_k_for_compression_ratio(-1, 1000, 128)
        print("✗ Negative ratio should have failed")
        return False
    except ValueError as e:
        print(f"✓ Negative ratio correctly failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Running SparQ fixes tests...\n")
    
    tests = [
        test_dense_attention,
        test_sparq_attention,
        test_invalid_sparq_ratio,
        test_compression_ratio_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.") 