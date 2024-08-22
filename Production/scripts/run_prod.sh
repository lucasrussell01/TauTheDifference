#!/bin/bash

# Preprocess
echo ""
python ../python/PreProcess.py
echo ""
echo "------------------------------------"
echo ""
# Shuffle and merge
python ../python/ShuffleMerge.py
echo ""
echo "------------------------------------"
echo ""
echo "Production complete!"