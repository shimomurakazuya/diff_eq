CXXFLAGS += -std=c++11 -O3 -Wall -Wextra
LDFLAGS += -std=c++11 -lm -Wall -Wextra

.PHONY: all

all: run
	
run: main.o ValuesDiffusion.o Output.o
	g++ -O3 -o $@ $^ $(LDFLAGS)

.cpp.o:
	g++ -O3 -c $< $(CXXFLAGS)
