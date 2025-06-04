/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 * @file   network_utils.h
 * @brief  network utilities
 */
#ifndef PS_NETWORK_UTILS_H_
#define PS_NETWORK_UTILS_H_
#include <unistd.h>
#ifdef _MSC_VER
#include <iphlpapi.h>
#include <tchar.h>
#include <windows.h>
#include <winsock2.h>
#undef interface
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#endif
#include <string>
#include <cstring>
#include <utility>
#include <array>
#include <vector>
#ifdef DMLC_USE_RDMA
#include <rdma/rdma_cma.h>
#endif
#ifdef DMLC_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "ps/internal/utils.h"
#include "ps/internal/postoffice.h"

namespace ps {

#ifdef DMLC_USE_CUDA
class NetDev {
 public:
  NetDev(struct ibv_context *context,
         int dev_id, struct ibv_device *dev,
         int port_id, struct ibv_port_attr port);
  ~NetDev();

  /**
   * Get best gid index
   * @return gid index
   */
  int get_best_gid_index();

  int get_best_gid(ibv_gid *gid, int *gid_idx);

  int get_port();

  int get_link();

  std::string get_name();

  char *get_pci_path();

  std::string get_ip();

  std::string get_interface_name();

  int read_pci_path();

 private:
  std::string dev_name_;
  ibv_context *context_ = nullptr;

  int dev_id_;
  uint8_t port_;
  uint8_t link_;
  char *pci_path_;
  int real_port_;
  int max_qp_num_;
  int gid_tbl_len_;

  int gid_idx_ = -1;
  ibv_gid gid_;
};
#endif

/**
 * \brief return the IP address for given interface eth0, eth1, ...
 */
static inline void GetIP(const std::string& interface, std::string* ip) {
#ifdef _MSC_VER
  typedef std::basic_string<TCHAR> tstring;
  // Try to get the Adapters-info table, so we can given useful names to the IP
  // addresses we are returning.  Gotta call GetAdaptersInfo() up to 5 times to
  // handle the potential race condition between the size-query call and the
  // get-data call. I love a well-designed API :^P
  IP_ADAPTER_INFO* pAdapterInfo = NULL;
  {
    ULONG bufLen = 0;
    for (int i = 0; i < 5; i++) {
      DWORD apRet = GetAdaptersInfo(pAdapterInfo, &bufLen);
      if (apRet == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);  // in case we had previously allocated it
        pAdapterInfo = static_cast<IP_ADAPTER_INFO*>(malloc(bufLen));
      } else if (apRet == ERROR_SUCCESS) {
        break;
      } else {
        free(pAdapterInfo);
        pAdapterInfo = NULL;
        break;
      }
    }
  }
  if (pAdapterInfo) {
    tstring keybase =
        _T(
        "SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\");
    tstring connection = _T("\\Connection");

    IP_ADAPTER_INFO* curpAdapterInfo = pAdapterInfo;
    while (curpAdapterInfo) {
      HKEY hKEY;
      std::string AdapterName = curpAdapterInfo->AdapterName;
      // GUID only ascii
      tstring key_set = keybase +
                        tstring(AdapterName.begin(), AdapterName.end()) +
                        connection;
      LPCTSTR data_Set = key_set.c_str();
      LPCTSTR dwValue = NULL;
      if (ERROR_SUCCESS ==
          ::RegOpenKeyEx(HKEY_LOCAL_MACHINE, data_Set, 0, KEY_READ, &hKEY)) {
        DWORD dwSize = 0;
        DWORD dwType = REG_SZ;
        if (ERROR_SUCCESS == ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType,
                                               (LPBYTE)dwValue, &dwSize)) {
          dwValue = new TCHAR[dwSize];
          if (ERROR_SUCCESS == ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType,
                                                 (LPBYTE)dwValue, &dwSize)) {
            // interface name must only ascii
            tstring tstr = dwValue;
            std::string s(tstr.begin(), tstr.end());
            if (s == interface) {
              *ip = curpAdapterInfo->IpAddressList.IpAddress.String;
              break;
            }
          }
        }
        ::RegCloseKey(hKEY);
      }
      curpAdapterInfo = curpAdapterInfo->Next;
    }
    free(pAdapterInfo);
  }
#else
  struct ifaddrs* ifAddrStruct = NULL;
  struct ifaddrs* ifa = NULL;
  void* tmpAddrPtr = NULL;

  getifaddrs(&ifAddrStruct);
  for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      // is a valid IP4 Address
      tmpAddrPtr =
          &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      if (strncmp(ifa->ifa_name, interface.c_str(), interface.size()) == 0) {
        *ip = addressBuffer;
        break;
      }
    }
  }
  if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
#endif
}

/**
 * \brief return the IP address and Interface the first interface which is not
 * loopback
 *
 * only support IPv4
 */
static inline void GetAvailableInterfaceAndIP(std::string* interface, std::string* ip) {
#ifdef _MSC_VER
  typedef std::basic_string<TCHAR> tstring;
  IP_ADAPTER_INFO* pAdapterInfo = NULL;
  {
    ULONG bufLen = 0;
    for (int i = 0; i < 5; i++) {
      DWORD apRet = GetAdaptersInfo(pAdapterInfo, &bufLen);
      if (apRet == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);  // in case we had previously allocated it
        pAdapterInfo = static_cast<IP_ADAPTER_INFO*>(malloc(bufLen));
      } else if (apRet == ERROR_SUCCESS) {
        break;
      } else {
        free(pAdapterInfo);
        pAdapterInfo = NULL;
        break;
      }
    }
  }
  if (pAdapterInfo) {
    tstring keybase =
        _T(
        "SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\");
    tstring connection = _T("\\Connection");

    IP_ADAPTER_INFO* curpAdapterInfo = pAdapterInfo;
    HKEY hKEY = NULL;
    while (curpAdapterInfo) {
      std::string curip(curpAdapterInfo->IpAddressList.IpAddress.String);
      if (curip == "127.0.0.1") {
        curpAdapterInfo = curpAdapterInfo->Next;
        continue;
      }
      if (curip == "0.0.0.0") {
        curpAdapterInfo = curpAdapterInfo->Next;
        continue;
      }

      std::string AdapterName = curpAdapterInfo->AdapterName;
      // GUID only ascii
      tstring key_set = keybase +
                        tstring(AdapterName.begin(), AdapterName.end()) +
                        connection;
      LPCTSTR data_Set = key_set.c_str();
      LPCTSTR dwValue = NULL;
      if (ERROR_SUCCESS ==
          ::RegOpenKeyEx(HKEY_LOCAL_MACHINE, data_Set, 0, KEY_READ, &hKEY)) {
        DWORD dwSize = 0;
        DWORD dwType = REG_SZ;
        if (ERROR_SUCCESS == ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType,
                                               (LPBYTE)dwValue, &dwSize)) {
          dwValue = new TCHAR[dwSize];
          if (ERROR_SUCCESS == ::RegQueryValueEx(hKEY, _T("Name"), 0, &dwType,
                                                 (LPBYTE)dwValue, &dwSize)) {
            // interface name must only ascii
            tstring tstr = dwValue;
            std::string s(tstr.begin(), tstr.end());

            *interface = s;
            *ip = curip;
            break;
          }
        }
        ::RegCloseKey(hKEY);
        hKEY = NULL;
      }
      curpAdapterInfo = curpAdapterInfo->Next;
    }
    if (hKEY != NULL) {
      ::RegCloseKey(hKEY);
    }
    free(pAdapterInfo);
  }
#else
  struct ifaddrs* ifAddrStruct = nullptr;
  struct ifaddrs* ifa = nullptr;

  interface->clear();
  ip->clear();
  getifaddrs(&ifAddrStruct);
  for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
    if (nullptr == ifa->ifa_addr) continue;

    if (AF_INET == ifa->ifa_addr->sa_family &&
        0 == (ifa->ifa_flags & IFF_LOOPBACK)) {
      char address_buffer[INET_ADDRSTRLEN];
      void* sin_addr_ptr =
          &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

      *ip = address_buffer;
      *interface = ifa->ifa_name;

      break;
    }
  }
  if (nullptr != ifAddrStruct) freeifaddrs(ifAddrStruct);
  return;
#endif
}

/**
 * \brief return an available port on local machine
 *
 * only support IPv4
 * \return 0 on failure
 */
static inline int GetAvailablePort(int num_ports, std::array<int, 32>* ports) {
  int socks[32];
  int num_available_ports = 0;
  for (int i = 0; i < num_ports; ++i) {
    struct sockaddr_in addr;
    addr.sin_port =
        htons(0);  // have system pick up a random port available for me
    addr.sin_family = AF_INET;                 // IPV4
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set our addr to any interface
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
      perror("bind():");
      return 0;
    }
#ifdef _MSC_VER
    int addr_len = sizeof(struct sockaddr_in);
#else
    socklen_t addr_len = sizeof(struct sockaddr_in);
#endif

    if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
      perror("getsockname():");
      return 0;
    }

    int ret_port = ntohs(addr.sin_port);
    ports->at(i) = ret_port;
    socks[i] = sock;
    num_available_ports += 1;
  }
  for (int i = 0; i < num_available_ports; ++i) {
    int sock = socks[i];
#ifdef _MSC_VER
    closesocket(sock);
#else
    close(sock);
#endif
  }
  return num_available_ports;
}

/**
 * \brief return the IP address and Interface based on current GPU
 * \return 0 on failure or no cuda, 1 when getting the interface successfully
 */
static inline int GetInterfaceAndIPByCurrentGpu(std::string* interface, std::string* ip) {
  interface->clear();
  ip->clear();

#ifdef DMLC_USE_CUDA
  int gpu = -1;
  cudaGetDevice(&gpu);
  if (gpu == -1) return 0;
  char pciPath[512];
  cudaDeviceProp deviceProp = {};
  cudaGetDeviceProperties(&deviceProp, gpu);
  snprintf(pciPath, sizeof(pciPath),
           "/sys/class/pci_bus/0000:%02x/device/0000:%02x:%02x.0/device",
           deviceProp.pciBusID, deviceProp.pciBusID, deviceProp.pciDeviceID);
  char* path = realpath(pciPath, nullptr);

  if (path == nullptr) return 0;

  int maxGidIdx = -1;

  struct ibv_device** devices;
  int numDev = 0;
  devices = ibv_get_device_list(&numDev);

  if (numDev <= 0) return 0;

  std::vector<NetDev*> devs;
  struct ibv_context *context = nullptr;
  for (int d = 0; d < numDev; d++) {
    if (nullptr != context && 0 != ibv_close_device(context)) {
      return 0;
    }
    context = ibv_open_device(devices[d]);
    if (context == nullptr) {
      continue;
    }

    struct ibv_device_attr attr = {};
    if (0 != ibv_query_device(context, &attr)) {
      continue;
    }

    for (int port = 1; port <= attr.phys_port_cnt; port++) {
      struct ibv_port_attr port_attr = {};
      if (0 != ibv_query_port(context, port, &port_attr)) {
        continue;
      }
      if (port_attr.state != IBV_PORT_ACTIVE) {
        continue;
      }
      if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND
          && port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        continue;
      }

      auto dev = new NetDev(context, d, devices[d], port, port_attr);
      if (dev->get_best_gid_index() > 0) {
        if (maxGidIdx < dev->get_best_gid_index()) {
          maxGidIdx = dev->get_best_gid_index();
        }
        devs.push_back(dev);
        // Then using net dev to release ibv context
        context = nullptr;
      }
    }
  }

  if (nullptr != context && 0 != ibv_close_device(context)) {
    PS_LOG(WARNING) << "failed to close device context, err=" << strerror(errno);
  }

  int maxPrefixLen = 0;
  int maxPrefixIdx = -1;
  for (uint32_t i = 0; i < devs.size(); i++) {
    int prefixLen = 0;
    auto dev = devs[i];
    if (dev->get_best_gid_index() < maxGidIdx) {
      continue;
    }

    auto net_pci_path = dev->get_pci_path();
    if (net_pci_path == nullptr) {
      continue;
    }

    while (path[prefixLen] != '\0'
           && net_pci_path[prefixLen] != '\0'
           && net_pci_path[prefixLen] == path[prefixLen]
           && prefixLen < 512) {
      prefixLen++;
    }
    if (prefixLen >= maxPrefixLen) {
      maxPrefixIdx = static_cast<int>(i);
      maxPrefixLen = prefixLen;
    }
  }

  ibv_free_device_list(devices);

  if (maxPrefixIdx == -1) {
    for (auto dev : devs) {
      free(dev);
    }
    return 0;
  }

  *ip = devs[maxPrefixIdx]->get_ip();
  *interface = devs[maxPrefixIdx]->get_interface_name();

  for (auto dev : devs) {
    free(dev);
  }

  return 1;
#else
  return 0;
#endif
}

}  // namespace ps
#endif  // PS_NETWORK_UTILS_H_
