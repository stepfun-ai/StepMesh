/**
 * Copyright (c) 2016 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_ENV_H_
#define PS_INTERNAL_ENV_H_
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
namespace ps {

/**
 * \brief Environment configurations
 */
class Environment {
 public:
  /**
   * \brief return the singleton instance
   */
  static inline Environment* Get() { return _GetSharedRef(nullptr).get(); }
  /**
   * \brief return a shared ptr of the singleton instance
   */
  static inline std::shared_ptr<Environment> _GetSharedRef() {
    return _GetSharedRef(nullptr);
  }
  /**
   * \brief initialize the environment
   * \param envs key-value environment variables
   * \return the initialized singleton instance
   */
  static inline Environment* Init(
      const std::unordered_map<std::string, std::string>& envs) {
    Environment* env = _GetSharedRef(&envs).get();
    env->kvs = envs;
    return env;
  }

  /**
   * \brief find the env value.
   *  User-defined env vars first. If not found, check system's environment
   * \param k the environment key
   * \return the related environment value, nullptr when not found
   */
  const char* find(const char* k) {
    std::string key(k);
    return kvs.find(key) == kvs.end() ? getenv(k) : kvs[key].c_str();
  }

  void set(const char* k, const std::string& rewrite_val) {
    std::string key(k);
    kvs[key] = std::move(rewrite_val);
  }

  /**
   * \brief find the env value with the integer type with a default value.
   *
   * \param k the environment key
   * \param val the pointer to the environment result with the integer type
   * \param default_val the default environment result when the environment
   *    key is not found
   * \return the integer value or the default value
   */
  int find(const char* k, int* val, int default_val = -1) {
    std::string key(k);
    auto val_str = kvs.find(key) == kvs.end() ? getenv(k) : kvs[key].c_str();
    if (val_str == nullptr) {
      *val = default_val;
      return -1;
    } else {
      *val = atoi(val_str);
      return 0;
    }
  }

  /**
   * \brief find the env value with the string type with a default value.
   *
   * \param k the environment key
   * \param val the pointer to the environment result with the integer type
   * \param default_val the default environment result when the environment
   *    key is not found
   * \return the string value
   */
  std::string find(const char* k, std::string& default_value) {
    std::string key(k);
    if (kvs.find(key) != kvs.end()) {
      return kvs[key];
    }

    if (getenv(k) == nullptr) {
      return default_value;
    }

    return {getenv(k)};
  }

 private:
  explicit Environment(
      const std::unordered_map<std::string, std::string>* envs) {
    if (envs) kvs = *envs;
  }

  static std::shared_ptr<Environment> _GetSharedRef(
      const std::unordered_map<std::string, std::string>* envs) {
    static std::shared_ptr<Environment> inst_ptr(new Environment(envs));
    return inst_ptr;
  }

  std::unordered_map<std::string, std::string> kvs;
};

}  // namespace ps
#endif  // PS_INTERNAL_ENV_H_
