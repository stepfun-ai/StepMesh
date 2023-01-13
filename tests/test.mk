TEST_SRC = $(wildcard tests/test_*.cc)
TEST = $(patsubst tests/test_%.cc, tests/test_%, $(TEST_SRC))

STATIC_LIBS = build/libps.a
ifeq ($(USE_TP), 1)
STATIC_LIBS += $(TP_INSTALL_PATH)/lib64/libtensorpipe.a $(TP_INSTALL_PATH)/lib64/libtensorpipe_uv.a
endif

# -ltcmalloc_and_profiler
LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -ldl
tests/% : tests/%.cc $(STATIC_LIBS)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/$* $< >tests/$*.d $(LIBS)
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(LDFLAGS) $(LIBS)

-include tests/*.d
