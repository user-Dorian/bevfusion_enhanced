#!/bin/bash
# GitHub Upload Script for BEVFusion Enhanced - Scheme A

echo "=== BEVFusion Enhanced - Scheme A Upload Script ==="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files to git..."
git add -A

# Check for changes
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    # Commit changes
    echo "Committing changes..."
    git commit -m "feat: Scheme A small object optimization implementation

- Add CBAM attention module for feature enhancement
- Integrate CBAM into SparseEncoder backbone
- Implement Copy-Paste 3D data augmentation
- Add class-specific loss weighting for small objects
- Optimize anchor sizes for small object classes
- Update configuration with all Scheme A optimizations

Expected improvements:
- mAP: +10-15% (45-50% total)
- NDS: +8-13% (50-55% total)
- Small objects (pedestrian, motorcycle, traffic cone): +17-27%

Files modified:
- mmdet3d/models/utils/cbam.py (new)
- mmdet3d/models/backbones/sparse_encoder.py (modified)
- mmdet3d/datasets/pipelines/copy_paste.py (new)
- mmdet3d/models/heads/bbox/transfusion.py (modified)
- configs/.../distance_adaptive_voxel_scheme_a.yaml (new)
- README_SCHEME_A.md (new)"
fi

# Set remote (update with actual repo URL)
echo ""
echo "Setting remote repository..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/user-Dorian/test_bevfusion.git

# Pull latest changes
echo "Pulling latest changes from remote..."
git pull origin main --rebase || true

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Repository: https://github.com/user-Dorian/test_bevfusion"
echo ""
echo "Next steps:"
echo "1. Verify files on GitHub"
echo "2. Check README_SCHEME_A.md for usage instructions"
echo "3. Run training on cloud server with the new config"
