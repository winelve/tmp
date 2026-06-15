#pragma once

#include "../extensions.h"

torch::Tensor random_walk_sdaa(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor start, int64_t walk_length);
