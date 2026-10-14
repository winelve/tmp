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
# - test/test_add.py
# - test/test_cat.py
# - test/test_coalesce.py
# - test/test_convert.py
# - test/test_diag.py
# - test/test_ego_sample.py
# - test/test_eye.py
# - test/test_matmul.py
# - test/test_metis.py
# - test/test_mul.py
# - test/test_neighbor_sample.py
# - test/test_overload.py
# - test/test_permute.py
# - test/test_saint.py
# - test/test_sample.py
# - test/test_spmm.py
# - test/test_spspmm.py
# - test/test_storage.py
# - test/test_tensor.py
# - test/test_transpose.py

pytest test/test_add.py test/test_cat.py test/test_coalesce.py \
       test/test_convert.py test/test_diag.py test/test_ego_sample.py \
       test/test_eye.py test/test_matmul.py test/test_metis.py \
       test/test_mul.py test/test_neighbor_sample.py test/test_overload.py \
       test/test_permute.py test/test_saint.py test/test_sample.py \
       test/test_spmm.py test/test_spspmm.py test/test_storage.py \
       test/test_tensor.py test/test_transpose.py -v "$@"

echo ""
echo "======================================"
echo "SDAA unit tests completed"
echo "======================================"
