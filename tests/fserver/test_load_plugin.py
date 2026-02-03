import torch, os, sys
import time

old_flags = sys.getdlopenflags()
sys.setdlopenflags(sys.getdlopenflags() | 0x100) 
import fserver_lib as f
sys.setdlopenflags(old_flags)

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'

f.init("libklx_backend.so")