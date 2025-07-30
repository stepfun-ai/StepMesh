/**
 *  Copyright (C) 2025 by StepAI Contributors
 */

#include "network_utils.h"  // NOLINT

#include <cstdio>
#include <sstream>
#include <string>

namespace ps {

#ifdef DMLC_USE_RDMA
NetDev::NetDev(struct ibv_context* context, int dev_id, struct ibv_device* dev,
               int port_id, struct ibv_port_attr port)
    : dev_name_(dev->name),
      context_(context),
      dev_id_(dev_id),
      port_(port_id),
      link_(port.link_layer),
      pci_path_(nullptr),
      gid_tbl_len_(port.gid_tbl_len) {
  read_pci_path();
  get_best_gid_index();
}

NetDev::~NetDev() {
  if (pci_path_ != nullptr) {
    free(pci_path_);
    pci_path_ = nullptr;
  }

  if (context_ != nullptr) {
    ibv_close_device(context_);
    context_ = nullptr;
  }
}

int NetDev::get_port() { return port_; }

int NetDev::get_link() { return link_; }

std::string NetDev::get_name() { return dev_name_; }

char* NetDev::get_pci_path() { return pci_path_; }

int NetDev::read_pci_path() {
  char device_path[1024];
  snprintf(device_path, sizeof(device_path), "/sys/class/infiniband/%s/device",
           dev_name_.c_str());
  char* p = realpath(device_path, nullptr);
  pci_path_ = p;
  if (p == nullptr) {
    return -1;
  }
  return 0;
}

int NetDev::get_best_gid_index() {
  if (gid_idx_ > 0) {
    return gid_idx_;
  }
  int gid_idx = -1;
  ibv_gid gid = {};
  for (int i = 1; i < gid_tbl_len_; i++) {
    bzero(&gid, sizeof(gid));
    ibv_query_gid(context_, port_, i, &gid);
    if ((gid.raw[10] == 0xFF) && (gid.raw[11] == 0xFF)) {
      gid_idx = i;
      gid_idx_ = i;
      memcpy(&gid_, &gid, sizeof(gid));
    }
  }
  return gid_idx;
}

int NetDev::get_best_gid(ibv_gid* gid, int* gid_idx) {
  if (gid_idx_ > 0) {
    *gid_idx = gid_idx_;
    memcpy(gid, &gid_, sizeof(gid_));
    return 0;
  }

  ibv_gid tmp_gid = {};

  for (int i = 1; i < gid_tbl_len_; i++) {
    bzero(&tmp_gid, sizeof(tmp_gid));
    ibv_query_gid(context_, port_, i, &tmp_gid);
    if ((tmp_gid.raw[10] == 0xFF) && (tmp_gid.raw[11] == 0xFF)) {
      *gid_idx = i;
      gid_idx_ = i;
      memcpy(&gid_, &tmp_gid, sizeof(tmp_gid));
      memcpy(gid, &tmp_gid, sizeof(tmp_gid));
    }
  }
  if (*gid_idx == -1) {
    return -1;
  }
  return 0;
}

std::string NetDev::get_ip() {
  std::stringstream ip;
  ip << static_cast<int>(gid_.raw[12]) << "." << static_cast<int>(gid_.raw[13])
     << "." << static_cast<int>(gid_.raw[14]) << "."
     << static_cast<int>(gid_.raw[15]);
  return ip.str();
}

std::string NetDev::get_interface_name() {
  int found = 0;
  struct ifaddrs *interfaces, *intf;
  getifaddrs(&interfaces);

  auto ip = get_ip();
  std::string name = "";
  for (intf = interfaces; intf && found < 32; intf = intf->ifa_next) {
    if (intf->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = intf->ifa_addr->sa_family;
    if (family != AF_INET) {
      continue;
    }

    std::string intf_ip = inet_ntoa(
        reinterpret_cast<struct sockaddr_in*>(intf->ifa_addr)->sin_addr);
    if (intf_ip == ip) {
      name = intf->ifa_name;
      break;
    }
  }

  freeifaddrs(interfaces);
  return name;
}

#endif  // DMLC_USE_RDMA
}  // namespace ps
