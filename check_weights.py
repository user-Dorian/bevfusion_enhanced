import torch
import sys

def main():
    checkpoint_path = 'train_result/latest.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  {key}")
        
        if 'state_dict' in checkpoint:
            print("\nState dict keys (first 20):")
            state_dict = checkpoint['state_dict']
            for i, key in enumerate(list(state_dict.keys())[:20]):
                print(f"  {key}")
            
            # Check fuser related keys
            print("\nFuser related keys:")
            fuser_keys = [k for k in state_dict.keys() if 'fuser' in k]
            for key in fuser_keys:
                print(f"  {key}")
        
        # Also check meta info
        if 'meta' in checkpoint:
            print(f"\nMeta info: {checkpoint['meta'].keys()}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()