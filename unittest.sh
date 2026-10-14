#!/bin/bash
# Unit test script for SDAA backend
# This script runs all tests that support the SDAA backend

set -e

echo "======================================"
echo "Running SDAA backend unit tests"
echo "======================================"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please install it with: pip install pytest"
    exit 1
fi

# Check if SDAA is available
python -c "import torch; assert hasattr(torch, 'sdaa') and torch.sdaa.is_available(), 'SDAA not available'" 2>/dev/null || {
    echo "Warning: SDAA is not available. Tests will be skipped or run on CPU only."
}

echo ""
echo "Running tests with SDAA support..."
echo ""

# Run tests that support SDAA backend
# Currently supported tests:
# - test/ops/test_add.py
# - test/ops/test_cat.py
# - test/ops/test_coalesce.py
# - test/ops/test_convert.py
# - test/ops/test_diag.py
# - test/ops/test_ego_sample.py
# - test/ops/test_eye.py
# - test/ops/test_matmul.py
# - test/ops/test_metis.py
# - test/ops/test_mul.py
# - test/ops/test_neighbor_sample.py
# - test/ops/test_overload.py
# - test/ops/test_permute.py
# - test/ops/test_saint.py
# - test/ops/test_sample.py
# - test/ops/test_spmm.py
# - test/ops/test_spspmm.py
# - test/ops/test_storage.py
# - test/ops/test_tensor.py
# - test/ops/test_transpose.py

pytest test/ops/test_add.py test/ops/test_cat.py test/ops/test_coalesce.py \
       test/ops/test_convert.py test/ops/test_diag.py test/ops/test_ego_sample.py \
       test/ops/test_eye.py test/ops/test_matmul.py test/ops/test_metis.py \
       test/ops/test_mul.py test/ops/test_neighbor_sample.py test/ops/test_overload.py \
       test/ops/test_permute.py test/ops/test_saint.py test/ops/test_sample.py \
       test/ops/test_spmm.py test/ops/test_spspmm.py test/ops/test_storage.py \
       test/ops/test_tensor.py test/ops/test_transpose.py -v "$@"

echo ""
echo "======================================"
echo "SDAA unit tests completed"
echo "======================================"

