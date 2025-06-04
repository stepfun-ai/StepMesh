/**
 *  Copyright (c) 2015 by Step AI
 */

#include "ut_common.h"

int main(int argc, char *argv[]) {
  StartPS(0, Node::SCHEDULER, -1, true);
  Finalize(0, Node::SCHEDULER, true);
  return 0;
}
