CXXFLAGS += -std=c++11 -O3 -Wall -Wextra
LDFLAGS += -std=c++11 -lm -Wall -Wextra
OMPFLAGS += -fopenmp

.PHONY: all clean resultclean tagfiles

TARGET := run

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
		g++ -O3 -o $@ $^ $(LDFLAGS) $(OMPFLAGS)

.cpp.o:
		g++ -O3 -c $< $(CXXFLAGS) $(OMPFLAGS)

clean:
		rm -f $(TARGET)
			rm -f $(OBJS)	
			rm -f data/*.dat

retultclean:
		find -name data/ '*.dat' | xargs rm -f 

tagfiles:
		ctags -R --langmap=c:+.hpp .
