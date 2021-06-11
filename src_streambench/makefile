CXXFLAGS += -std=c++11 -fopenmp -O3 -Wall -Wextra
LDFLAGS += -std=c++11 -fopenmp -lm -Wall -Wextra


.PHONY: all clean resultclean tagfiles
 
 TARGET := run
 
 SRCS := $(wildcard *.cpp)
 OBJS := $(SRCS:.cpp=.o)


all: run

run: main.o
	g++ -O3 -o $@ $^ $(LDFLAGS)
#gcc -O3 -o $@ $^ $(LDFLAGS)

.c.cpp.o:
	g++ -O3 -c $< $(CXXFLAGS)
#gcc -O3 -c $< $(CXXFLAGS)

clean:
	rm -f $(TARGET)
	rm -f $(OBJS)   
	rm -f *.e* 
	rm -f *.o* 

resultclean:
	find -name data/ '*.dat' | xargs rm -f 
	rm -f dif2d.*


tagfiles:
	ctags -R --langmap=c:+.hpp .



