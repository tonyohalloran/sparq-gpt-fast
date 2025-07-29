#!/usr/bin/env python3
"""
Simple test to verify SparQ fixes.
"""

import torch
from sparq import get_r_k_for_compression_ratio, SparQArgs, RKForCompressionRatio

def test_compression_ratio():
    """Test compression ratio validation."""
    print("Testing compression ratio validation...")
    
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
        print("✗ Negative ratio -1 should have failed")
        return False
    except ValueError as e:
        print(f"✓ Negative ratio -1 correctly failed: {e}")
    
    return True

def test_sparq_args():
    """Test SparQArgs validation."""
    print("\nTesting SparQArgs validation...")
    
    # Test valid args
    try:
        args = SparQArgs(rk=RKForCompressionRatio(ratio=8))
        print("✓ Valid SparQArgs created")
    except Exception as e:
        print(f"✗ Valid SparQArgs failed: {e}")
        return False
    
    # Test invalid args
    try:
        args = SparQArgs(rk=RKForCompressionRatio(ratio=0))
        print("✗ Invalid SparQArgs should have failed")
        return False
    except Exception as e:
        print(f"✓ Invalid SparQArgs correctly failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Running simple SparQ fix tests...\n")
    
    success1 = test_compression_ratio()
    success2 = test_sparq_args()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!") 