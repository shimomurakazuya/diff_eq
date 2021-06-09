CXXFLAGS += -std=c++11 -O3 -Wall -Wextra
LDFLAGS += -std=c++11 -lm -Wall -Wextra
OMPFLAGS += -fopenmp

.PHONY: all clean resultclean tagfiles

TARGET := run

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
		icpc -O3 -xCORE-AVX512 -o $@ $^ $(LDFLAGS) -qopenmp
####icpc -O2 -xCORE-AVX2  -o $@ $^ $(LDFLAGS) -qopenmp

.cpp.o:
		icpc -O3 -xCORE-AVX512  -c $< $(CXXFLAGS) -qopenmp
###icpc -O2 -xCORE-AVX2  -c $< $(CXXFLAGS) -qopenmp
	

clean:
		rm -f $(TARGET)
			rm -f $(OBJS)	

resultclean:
		find -name data/ '*.dat' | xargs rm -f 
			rm -f dif2d.*

tagfiles:
		ctags -R --langmap=c:+.hpp .
