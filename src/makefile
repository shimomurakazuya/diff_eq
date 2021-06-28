

CXX := nvcc
#CXX := g++

ifeq ("$(CXX)", "nvcc")   # CXX as cuda
CXXFLAGS := -x cu
CXXFLAGS += -arch=sm_70 # sm_70 == Volta generation
CXXFLAGS += -use_fast_math
CXXFLAGS += -std=c++11 -O3  -Xcompiler=" -Wall -Wextra"
CXXFLAGS += -Xcompiler="-fopenmp"
CXXFLAGS += -extended-lambda
CXXFLAGS += -D USE_NVCC
#CXXFLAGS += -D DEBUG
endif
ifeq ("$(CXX)", "g++")    #CXX as OMP
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra
CXXFLAGS += -D USE_OMP
endif

LD := $(CXX)
ifeq ("$(CXX)", "nvcc")   # CXX as cuda
LDFLAGS := -std=c++17 -lm
LDFLAGS += -Xcompiler="-fopenmp"
endif
ifeq ("$(CXX)", "g++")    #CXX as OMP
LDFLAGS := -std=c++17 -lm -fopenmp -Wall -Wextra
endif


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
	find data/ -name '*.dat' | xargs rm -f
	rm -f dif2d.*

tagfiles:
	ctags -R --langmap=c:+.hpp .
