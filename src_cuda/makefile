# CXX as cuda
CXX := nvcc
CXXFLAGS := -x cu
CXXFLAGS += -arch=sm_70 # sm_70 == Volta generation
CXXFLAGS += -use_fast_math
CXXFLAGS += -std=c++17 -O3 -Xcompiler="-Wall -Wextra"
#CXXFLAGS += -Xcompiler="-fopenmp"

LD := $(CXX)
LDFLAGS := -std=c++17 -lm
#LDFLAGS += -Xcompiler="-fopenmp"

.PHONY: all clean resultclean tagfiles
TARGET := run
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) -o  $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f $(TARGET)
	rm -f $(OBJS)	

resultclean:
	find -name data/ '*.dat' | xargs rm -f
	rm -f dif2d.*

tagfiles:
	ctags -R --langmap=c:+.hpp .
