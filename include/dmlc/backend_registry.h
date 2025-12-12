#pragma once

#include "base.h"
#include "ps/backend.h"

namespace dmlc {

// StepMesh Backend Registration
//
// StepMesh uses a **dynamic backend registration mechanism** to decouple
// specific backend implementations (e.g., RDMA, CPU, GPU, future XPUs) from
// the core transport layer.
//
// Motivation:
//   * Avoid hard-coding backend logic into the core codebase.
//   * Support extensibility: new backends can be plugged in without touching
//     core StepMesh logic.
//   * Enable backend discovery at runtime via a registry pattern.
//
// Contract Requirements:
//   * Each backend must define a unique type identifier.
//   * Backends must satisfy the BackendInterface contract (init, push/pull, etc).
//   * Registration should occur during StepMesh startup or module load.
//
// Backend registry responsibilities:
//   * Store mappings from backend ID to constructor/factory.
//   * Provide factory APIs to create instances by ID.
//   * Ensure correct initialization order (core first, backends next).
//
// Example use cases:
//   * Enable support for new accelerator types (e.g., NPU, TPU-like devices).
//   * Abstract transport layer differences while exposing uniform APIs.
//
// Note: This refactor **does not add a new mandatory backend** — existing
// backends continue to work without modification unless explicitly replaced.
// It simply provides the structural foundation for extensibility.
// This design choice is focused on modularity and future-proofing.
template <typename T>
struct STEPMESH_API backend_registry {
  backend_registry(const std::string& name) {
    ps::Backend::RegisterLazy(name, []() { return new T(); });
  }
};

}  // namespace dmlc
