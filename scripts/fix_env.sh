#!/bin/bash
# Run this once on HPC to fix the transformers/torch version conflict
# torch 2.12.0 broke the tensor materialization path in transformers 5.x
# Pin to the last known-good version pair

pip install "transformers==4.46.3" --break-system-packages
pip install "tokenizers==0.20.3" --break-system-packages
pip install "accelerate==1.2.1" --break-system-packages

echo "Done. Verify:"
python3 -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"
