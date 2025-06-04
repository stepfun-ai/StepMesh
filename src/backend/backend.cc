/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/backend.h"

namespace ps {

std::mutex Backend::backends_mutex_;
std::unordered_map<std::string, Backend*> Backend::backends_;

}  // namespace ps