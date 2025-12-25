/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/backend.h"

#include <string>
#include <unordered_map>

namespace ps {

std::mutex Backend::backends_mutex_;
std::unordered_map<std::string, Backend*> Backend::backends_;
std::unordered_map<std::string, std::function<Backend*(void)>>
    Backend::backend_ctors_;

void Backend::RegisterLazy(const std::string& name,
                           const std::function<Backend*(void)>& ctor) {
  Backend::backend_ctors_.emplace(name, ctor);
}

}  // namespace ps
