# NiceMPI
An alternative to Boost.MPI for a user friendly C++ interface for MPI (MPICH).

# Rationale

The main advantage of this library when compared to other C++ MPI wrapper that I know about is that it does not require to *register* user-defined types with a MPI facility like `MPI_Type_*`. This is true for any so-called [POD](http://en.cppreference.com/w/cpp/concept/PODType) type. To achieve this, we used the fact that, by definition, an unsigned char in C++ is a [byte](https://en.wikipedia.org/wiki/Byte) and that, according to the [C++11 standard draft](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3337.pdf)

> A pointer to a standard-layout struct object, suitably converted using a reinterpret_cast, points to its
> initial member (or if that member is a bit-field, then to the unit in which it resides) and vice versa. [ Note:
> There might therefore be unnamed padding within a standard-layout struct object, but not at its beginning,
as necessary to achieve appropriate alignment. - end note ]

Hence, internally, all the communications with MPI in this library

1. First make sure that the type that is manipulated is indeed [POD](http://en.cppreference.com/w/cpp/concept/PODType) by using [`std::is_pod`](http://en.cppreference.com/w/cpp/types/is_pod). 
2. Treat the type as an array of bytes (unsigned char).

Performances are a legitimate concern with this approach. What if, internally, the MPI implementation is able to use the knowledge of the data type to perform some optimization? However, a quick look at [MPICH](https://www.mpich.org/) source hints that optimizations seem to depends only on data size, not on data type. If this information were found to be invalid, I believe that this performance issue could be fixed and that the interface of this library could still remain intact at the expense of a more complicated implementation.

# Dependencies

- A C++ compiler that supports C++11, like any somewhat recent versions of [clang++](http://clang.llvm.org/) or [g++](https://gcc.gnu.org/)
- [MPICH](https://www.mpich.org/), an open source implementation of MPI. Other implementations of MPI might work with this library, but they were not tested.

# Usage

The typical program that print the world size and the rank of each process looks like

```c++
#include <iostream>
#include <NiceMPI.h>

int main(int argc, char* argv[]) {
	using namespace NiceMPI;
	MPI_RAII instance{argc,argv};
	if(getWorld().rank() == 0) {
		std::cout << "The world size is " <<  getWorld().size() << std::endl;
	}
	std::cout << "I have rank " << getWorld().rank() << std::endl;
	return 0;
}
```

This program can be ran on 8 cores with *mpiexec -np 8 theProgramName*. The ``MPI_RAII`` struct initialize and finalize MPI using the [RAII programming idiom](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization). ``getWorld()`` is a function that returns a global ``Communicator``. The class ``Communicator`` will be described in more details later. In the present context, it suffices to know that the ``Communicator`` returned by ``getWorld()`` is a wrapper around ``MPI_COMM_WORLD``.

Once you have a communicator, sending and receiving data is very easy. For instance, the code to send the char *'K'* from the first process in the world to the last is

```c++
const int sourceIndex = 0;
const int destinationIndex = getWorld().size() -1;
const unsigned char toSend = 'K';
if(getWorld().rank() == sourceIndex) {
	getWorld().sendDataTo(toSend,destinationIndex);
}
if(getWorld().rank() == destinationIndex) {
	auto result = getWorld().receiveDataFrom<unsigned char>(sourceIndex);
}
```

As advertised, it is very easy to go from this example to an example where you send any [POD](http://en.cppreference.com/w/cpp/concept/PODType) type.

```c++
struct MyStruct {
	double a;
	int b;
	char c;
};
const MyStruct toSend{6.66,42,'K'};
if(getWorld().rank() == sourceIndex) {
	getWorld().sendDataTo(toSend,destinationIndex);
}
if(getWorld().rank() == destinationIndex) {
	auto result = getWorld().receiveDataFrom<MyStruct>(sourceIndex);
}
```

Typical MPI functions are implemented, and they can all be used with [POD](http://en.cppreference.com/w/cpp/concept/PODType)

```c++
MyStruct broadcasted = getWorld().broadcast(sourceIndex, toSend);
const int sendCount = 2;
std::vector<MyStruct> vecToSend(sendCount*getWorld().size());
std::vector<MyStruct> scattered = getWorld().scatter(sourceIndex, vecToSend, sendCount);
```
