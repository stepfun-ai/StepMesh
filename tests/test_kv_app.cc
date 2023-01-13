#include <cmath>
#include "ps/ps.h"

using namespace ps;

void StartServer() {
  if (!IsServer()) {
    return;
  }
  auto server = new KVServer<float>(0);
  server->set_request_handle(KVServerDefaultHandle<float>());
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (IsServer() || IsScheduler()) return;
  KVWorker<float> kv(0, 0);

  // init
  int num = 10000;
  SArray<Key> keys(num);
  SArray<float> vals(num);

  int rank = MyRank();
  srand(rank + 7);
  for (int i = 0; i < num; ++i) {
    keys[i] = kMaxKey / num * i + rank;
    vals[i] = (rand() % 1000);
  }

  // push
  int repeat = 50;
  std::vector<int> ts;
  for (int i = 0; i < repeat; ++i) {
    ts.push_back(kv.ZPush(keys, vals));

    // to avoid too frequency push, which leads huge memory usage
    if (i > 10) kv.Wait(ts[ts.size()-10]);
  }
  for (int t : ts) kv.Wait(t);

  // pull
  SArray<float> rets;
  kv.Wait(kv.ZPull(keys, &rets));

  float res = 0;
  for (int i = 0; i < num; ++i) {
    res += std::fabs(rets[i] - vals[i] * repeat);
  }
  CHECK_LT(res / repeat, 1e-5);
  LL << "error: " << res / repeat;
}

int main(int argc, char *argv[]) {
  const char* val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  std::string role_str(val);
  Node::Role role = GetRole(role_str);
  // start system
  StartPS(0, role, -1, true);
  // setup server nodes
  StartServer();
  // run worker nodes
  RunWorker();
  // stop system
  Finalize(0, role, true);
  return 0;
}
