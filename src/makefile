CXXFLAGS += -std=c++11 -fopenmp -O3 -Wall -Wextra
LDFLAGS += -std=c++11 -fopenmp -lm -Wall -Wextra

.PHONY: all

all: run
	
run: main.o
	g++ -O3 -o $@ $^ $(LDFLAGS)
#	gcc -O3 -o $@ $^ $(LDFLAGS)

.cpp.o:
	g++ -O3 -c $< $(CXXFLAGS)
#	gcc -O3 -c $< $(CXXFLAGS)
