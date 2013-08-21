CC = g++
CFLAGS = -Iinclude -I/usr/local/include/eigen3

.cc.o:
	g++ $(CFLAGS) -o $*.o -c $*.cc

test: test.cc include/*.h
	g++ $(CFLAGS) test.cc -o test

clean:
	rm -rf test *.o
