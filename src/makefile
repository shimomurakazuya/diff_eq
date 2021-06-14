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
#gcc -O3 -o $@ $^ $(LDFLAGS)

.cpp.o:
	g++ -O3 -c $< $(CXXFLAGS) $(OMPFLAGS)
#gcc -O3 -c $< $(CXXFLAGS)



clean:
		rm -f $(TARGET)
			rm -f $(OBJS)	

resultclean:
		find -name data/ '*.dat' | xargs rm -f 
			rm -f dif2d.*

tagfiles:
		ctags -R --langmap=c:+.hpp .
