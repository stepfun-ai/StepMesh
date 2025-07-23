/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_MULTI_QP_H_
#define  PS_INTERNAL_MULTI_QP_H_

#include "ps/internal/utils.h"

// Limited by RDMA Write Inline Size
#define QP_MAX_NUM 2

static int QP_NUM = ps::GetEnv("STEPAF_QP_NUM", 2);  // Number of QPs

#define FOR_QPS for (int qpIndex = 0; qpIndex < QP_NUM; qpIndex++)

#endif  // PS_INTERNAL_MULTI_QP_H_
