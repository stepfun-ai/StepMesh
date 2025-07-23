#!/usr/bin/env bash
cpplint --root=./src --filter=-whitespace/indent_namespace,-runtime/references \
  --exclude=./src/zmq_van.h \
  --exclude=./src/windows/* \
  --exclude=./src/ucx_van.h \
  --exclude=./src/tp_van.h \
  --exclude=./src/resender.h \
  --exclude=./src/multi_van.h \
  --exclude=./src/fabric_van.h \
  --exclude=./src/fabric_utils.h \
  --exclude=./src/fabric_transport.h \
  --recursive ./src
cpplint --root=./include --filter=-whitespace/indent_namespace,-runtime/references \
  --exclude=./include/ps/internal/spsc_queue.h \
  --exclude=./include/ps/internal/parallel_sort.h \
  --exclude=./include/ps/internal/parallel_kv_match.h \
  --exclude=include/dmlc/* \
  --recursive ./include
cpplint --root=./fserver/csrc --filter=-whitespace/indent_namespace,-runtime/references \
  --recursive ./fserver