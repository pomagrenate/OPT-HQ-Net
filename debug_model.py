import torch
import torch.nn as nn
import sys

print("Debugging model step by step...")

try:
    from model import Conv, autopad, Bottleneck, C3k2, SPPF, FilamentSegmentation
    
    # Test 1: Basic Conv
    print("1. Testing Conv...")
    conv = Conv(32, 64, 3, 2)
    x = torch.randn(1, 32, 1024, 1024)
    y = conv(x)
    print(f"   Conv works: {x.shape} -> {y.shape}")
    
    # Test 2: SPPF
    print("2. Testing SPPF...")
    sppf = SPPF(256, 256, 5)
    x = torch.randn(1, 256, 64, 64)
    y = sppf(x)
    print(f"   SPPF works: {x.shape} -> {y.shape}")
    
    # Test 3: Backbone
    print("3. Testing backbone...")
    backbone = nn.ModuleList([
        Conv(1, 32, 3, 2),          # 0: 2048->1024
        Conv(32, 64, 3, 2),         # 1: 1024->512 (P2)
        Conv(64, 128, 3, 2),        # 2: 512->256 (P3)
        Conv(128, 192, 3, 2),       # 3: 256->128 (P4)
        Conv(192, 256, 3, 2),       # 4: 128->64 (P5)
        SPPF(256, 256, 5),          # 5
    ])
    
    x = torch.randn(1, 1, 256, 256)  # Start with smaller input
    for i, layer in enumerate(backbone):
        x = layer(x)
        print(f"   Layer {i}: {x.shape}")
    
    print("   Backbone works!")
    
    # Test 4: Full model with small input
    print("4. Testing full model with small input...")
    model = FilamentSegmentation(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"   Full model works: {x.shape} -> {y.shape}")
    
    print("\n[SUCCESS] All tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)