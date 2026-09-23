import torch
import sys

print("Testing new model architecture...")

try:
    from model import FilamentSegmentation, count_parameters
    
    # Test 1: Create model
    print("1. Creating model...")
    model = FilamentSegmentation(in_channels=1, num_classes=1)
    print(f"   Model created with {count_parameters(model):,} parameters")
    
    # Test 2: Forward pass with smaller input first
    print("2. Testing with 256x256 input...")
    dummy_small = torch.randn(1, 1, 256, 256)
    mask_small = model(dummy_small)
    print(f"   Small test passed: {mask_small.shape}")
    
    # Test 3: Forward pass with full resolution
    print("3. Testing with 2048x2048 input...")
    dummy_input = torch.randn(1, 1, 2048, 2048)
    mask = model(dummy_input)
    print(f"   Full resolution test passed: {mask.shape}")
    
    # Test 4: Two channel input
    print("4. Testing with two-channel input...")
    model_2ch = FilamentSegmentation(in_channels=2, num_classes=1)
    dummy_2ch = torch.randn(1, 2, 2048, 2048)
    mask_2ch = model_2ch(dummy_2ch)
    print(f"   Two-channel test passed: {mask_2ch.shape}")
    
    print("\n[SUCCESS] All tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)