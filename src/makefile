#CXXFLAGS += -std=c++17 -O3 -Wall -Wextra
#LDFLAGS  += -std=c++17 -lm -Wall -Wextra
CXXFLAGS += -std=c++17 -O3  
LDFLAGS  += -std=c++17 -lm  
#OMPFLAGS += -fopenmp
#NVCFLAGS += -g -O3 -std=c++11 -restrict
##NVCFLAGS += --expt-extended-lambda
##NVCFLAGS += -use_fast_math
##NVCFLAGS += --default-stream per-thread
#NVCFLAGS += -Xptxas -v -lineinfo
#NVCFLAGS += -maxrregcount 127
NVCFLAGS += --nvlink-options -Werror


.PHONY: all clean resultclean tagfiles

TARGET := run

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

  $(TARGET): $(OBJS)
	nvcc -O3 -o  $@ $^ $(LDFLAGS) $(OMPFLAGS) $(NVCFLAGS)


.cpp.o:
	nvcc -O3 -c  $< $(CXXFLAGS) $(OMPFLAGS) $(NVCFLAGS)


clean:
		rm -f $(TARGET)
			rm -f $(OBJS)	

resultclean:
		find -name data/ '*.dat' | xargs rm -f 
			rm -f dif2d.*

tagfiles:
		ctags -R --langmap=c:+.hpp .
