/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of dmlc
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef DMLC_LOGGING_H_
#define DMLC_LOGGING_H_
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include "./base.h"
#include "ps/internal/utils.h"

#if DMLC_LOG_STACK_TRACE
#include <cxxabi.h>
#endif

#if DMLC_LOG_STACK_TRACE
#include <execinfo.h>
#endif

namespace dmlc {
/*!
 * \brief exception class that will be thrown by
 *  default logger if DMLC_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept(a)
#endif

#if DMLC_USE_CXX11
#define DMLC_THROW_EXCEPTION noexcept(false)
#else
#define DMLC_THROW_EXCEPTION
#endif

#if DMLC_USE_GLOG
#include <glog/logging.h>

namespace dmlc {
inline void InitLogging(const char *argv0) { google::InitGoogleLogging(argv0); }
}  // namespace dmlc

#else
// use a light version of glog
#include <assert.h>
#include <ctime>
#include <iostream>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#endif

namespace dmlc {
inline void InitLogging(const char *argv0) {
  // DO NOTHING
}

// Always-on checking
#define PS_CHECK(x)                                                      \
  if (!(x))                                                           \
  dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check "      \
                                                        "failed: " #x \
                                                     << ' '
#define PS_CHECK_LT(x, y) PS_CHECK((x) < (y))
#define PS_CHECK_GT(x, y) PS_CHECK((x) > (y))
#define PS_CHECK_LE(x, y) PS_CHECK((x) <= (y))
#define PS_CHECK_GE(x, y) PS_CHECK((x) >= (y))
#define PS_CHECK_EQ(x, y) PS_CHECK((x) == (y))
#define PS_CHECK_NE(x, y) PS_CHECK((x) != (y))
#define PS_CHECK_NOTNULL(x)                                                                 \
  ((x) == NULL                                                                           \
   ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: " #x << ' ', \
   (x) : (x))  // NOLINT(*)
// Debug-only checking.
#ifdef NDEBUG
/* 
#define DPS_CHECK(x) \
  while (false) PS_CHECK(x)
#define DPS_CHECK_LT(x, y) \
  while (false) PS_CHECK((x) < (y))
#define DPS_CHECK_GT(x, y) \
  while (false) PS_CHECK((x) > (y))
#define DPS_CHECK_LE(x, y) \
  while (false) PS_CHECK((x) <= (y))
#define DPS_CHECK_GE(x, y) \
  while (false) PS_CHECK((x) >= (y))
#define DPS_CHECK_EQ(x, y) \
  while (false) PS_CHECK((x) == (y))
#define DPS_CHECK_NE(x, y) \
  while (false) PS_CHECK((x) != (y))
#else
#define DPS_CHECK(x) PS_CHECK(x)
#define DPS_CHECK_LT(x, y) PS_CHECK((x) < (y))
#define DPS_CHECK_GT(x, y) PS_CHECK((x) > (y))
#define DPS_CHECK_LE(x, y) PS_CHECK((x) <= (y))
#define DPS_CHECK_GE(x, y) PS_CHECK((x) >= (y))
#define DPS_CHECK_EQ(x, y) PS_CHECK((x) == (y))
#define DPS_CHECK_NE(x, y) PS_CHECK((x) != (y)) */
#endif  // NDEBUG

#define PS_LOG_API dmlc::LogMessage(__FILE__, __LINE__)

#define PS_LOG_IF(severity, condition) \
  !(condition) ? (void)0 : dmlc::LogMessageVoidify() & PS_LOG_API

#define LOG_FATAL dmlc::LogMessageFatal(__FILE__, __LINE__)
#define PS_LOG_FATAL LOG_FATAL.stream()
#define LOG_QFATAL LOG_FATAL

#define PS_VLOG(x) PS_LOG_IF(INFO, x <= PS_VERBOSE).stream()

#define PS_LOG_ERROR   PS_VLOG(0)
#define PS_LOG_FATAL   LOG_FATAL.stream()
#define PS_LOG_WARNING PS_VLOG(0)
#define PS_LOG_INFO    PS_VLOG(1)
#define PS_LOG_TRACE   PS_VLOG(2)
#define PS_LOG_ALL     PS_VLOG(3)

#define PS_LOG(severity) PS_LOG_##severity

#define PS_LG LOG_INFO.stream()

#ifdef NDEBUG
/*
#define LOG_DFATAL PS_LOG_ERROR
#define DFATAL ERROR
#define DLOG(severity) true ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#else
#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) PS_LOG_IF(severity, condition)
*/
#endif



// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char *HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm now;
    localtime_r(&time_value, &now);
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", now.tm_hour, now.tm_min, now.tm_sec);
#endif
    return buffer_;
  }

 private:
  char buffer_[9];
};

class LogMessage {
 public:
  LogMessage(const char *file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] "
                << getenv("DMLC_ROLE") << " " << getenv("STEPMESH_GPU") << " " << file << ":" << std::dec << line << ": ";
  }
  ~LogMessage() { log_stream_ << "\n"; }
  std::ostream &stream() { return log_stream_; }

 protected:
  std::ostream &log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage &);
  void operator=(const LogMessage &);
};

#if DMLC_LOG_STACK_TRACE
inline std::string Demangle(char const *msg_str) {
  using std::string;
  string msg(msg_str);
  size_t symbol_start = string::npos;
  size_t symbol_end = string::npos;
  if (((symbol_start = msg.find("_Z")) != string::npos) &&
      (symbol_end = msg.find_first_of(" +", symbol_start))) {
    string left_of_symbol(msg, 0, symbol_start);
    string symbol(msg, symbol_start, symbol_end - symbol_start);
    string right_of_symbol(msg, symbol_end);

    int status = 0;
    size_t length = string::npos;
    std::unique_ptr<char, decltype(&std::free)> demangled_symbol = {
        abi::__cxa_demangle(symbol.c_str(), 0, &length, &status), &std::free};
    if (demangled_symbol && status == 0 && length > 0) {
      string symbol_str(demangled_symbol.get());
      std::ostringstream os;
      os << left_of_symbol << symbol_str << right_of_symbol;
      return os.str();
    }
  }
  return string(msg_str);
}

inline std::string StackTrace() {
  using std::string;
  std::ostringstream stacktrace_os;
  const int MAX_STACK_SIZE = DMLC_LOG_STACK_TRACE_SIZE;
  void *stack[MAX_STACK_SIZE];
  int nframes = backtrace(stack, MAX_STACK_SIZE);
  stacktrace_os << "Stack trace returned " << nframes << " entries:" << std::endl;
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int frameno = 0; frameno < nframes; ++frameno) {
      string msg = dmlc::Demangle(msgs[frameno]);
      stacktrace_os << "[bt] (" << frameno << ") " << msg << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}

#else  // DMLC_LOG_STACK_TRACE is off

inline std::string demangle(char const *msg_str) { return std::string(); }

inline std::string StackTrace() {
  return std::string(
      "stack traces not available when "
      "DMLC_LOG_STACK_TRACE is disabled at compile time.");
}

#endif  // DMLC_LOG_STACK_TRACE

#if DMLC_LOG_FATAL_THROW == 0

class LogMessageNull :public LogMessage {
  public:
  LogMessageNull(){
    std::ostringstream oss;
    return oss;
  }

}

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char *file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n";
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal &);
  void operator=(const LogMessageFatal &);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char *file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":" << line << ": ";
  }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() DMLC_THROW_EXCEPTION {
#if DMLC_LOG_STACK_TRACE
    log_stream_ << "\n\n" << StackTrace() << "\n";
#endif
    // throwing out of destructor is evil
    // hopefully we can do it here
    // also log the message before throw
    PS_LOG_API.stream() << log_stream_.str();
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal &);
  void operator=(const LogMessageFatal &);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream &) {}
};

}  // namespace dmlc

#endif
#endif  // DMLC_LOGGING_H_
