CPP		= g++
CFLAGS		= -I../../libcuda/ -Wall -O3 -pthread -funroll-loops
LDFLAGS		= -L../../libcuda/ -lm5op_x86 -static
SOURCE		= cpu_memory_demand.cpp
EXECUTABLE	= cpu_memory_demand

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CPP) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)
