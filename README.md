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

Another choice made with this library is to support only C++11 and more. This significantly simplifies the implementation. It is plausible to think that this simplification may ultimately benifit the interface.

# Dependencies

- A C++ compiler that supports C++11, like recent versions of [clang++](http://clang.llvm.org/) or [g++](https://gcc.gnu.org/) [<sup>1</sup>](#footnoteOne). 
- [MPICH](https://www.mpich.org/), an open source implementation of MPI. Other implementations of MPI might work with this library, but they were not tested.

# Usage

The typical program that print the world size and the rank of each process looks like

```c++
#include <iostream>
#include <NiceMPI.h>

int main(int argc, char* argv[]) {
	using namespace NiceMPI;
	MPI_RAII instance{argc,argv};
	if(mpiWorld().rank() == 0) {
		std::cout << "The world size is " <<  mpiWorld().size() << std::endl;
	}
	std::cout << "I have rank " << mpiWorld().rank() << std::endl;
	return 0;
}
```

This program can be ran on 8 cores with *mpiexec -np 8 theProgramName*. The ``MPI_RAII`` struct initialize and finalize MPI using the [RAII programming idiom](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization). ``mpiWorld()`` is a function that returns a global ``Communicator``. The class ``Communicator`` will be described in more details later. In the present context, it suffices to know that the ``Communicator`` returned by ``mpiWorld()`` is a wrapper around ``MPI_COMM_WORLD``.

Once you have a communicator, sending and receiving data is very easy. For instance, the code to send the char *'K'* from the first process in the world to the last is

```c++
const int sourceIndex = 0;
const int destinationIndex = mpiWorld().size() -1;
const unsigned char toSend = 'K';
if(mpiWorld().rank() == sourceIndex) {
	mpiWorld().sendAndBlock(toSend,destinationIndex);
}
if(mpiWorld().rank() == destinationIndex) {
	auto result = mpiWorld().receiveAndBlock<unsigned char>(sourceIndex);
}
```

You'll notice that, when necessary, the wrapper use a more precise name than the MPI function it wraps. For instance, above, ``MPI_Send`` and ``MPI_Recv`` were renamed to ``sendAndBlock`` and ``receiveAndBlock`` in order to put the emphasis on the blocking property of these functions.

As advertised, it is very easy to go from this example to an example where you send any [POD](http://en.cppreference.com/w/cpp/concept/PODType) type

```c++
struct MyStruct {
	double a;
	int b;
	char c;
};
const MyStruct toSend{6.66,42,'K'};
if(mpiWorld().rank() == sourceIndex) {
	mpiWorld().sendAndBlock(toSend,destinationIndex);
}
if(mpiWorld().rank() == destinationIndex) {
	auto result = mpiWorld().receiveAndBlock<MyStruct>(sourceIndex);
}
```

Non blocking send and receive functions are also available

```c++
if(mpiWorld().rank() == sourceIndex) {
	SendRequest r = mpiWorld().asyncSend(toSend,destinationIndex);
	r.wait();
}
if(mpiWorld().rank() == destinationIndex) {
	ReceiveRequest<MyStruct> r = mpiWorld().asyncReceive<MyStruct>(sourceIndex);
	r.wait();
	std::unique_ptr<MyStruct> data = r.take();
}
```

The classes `SendRequest` and `ReceiveRequest` also implement the function `isCompleted()` that returns true if the request is completed, i.e. if the data were respectively sent or received.

Typical MPI functions are implemented, and they can all be used with [POD](http://en.cppreference.com/w/cpp/concept/PODType). For instance, the basic collective communication methods are

```c++
//Useless examples since every process got all the data...
MyStruct broadcasted = mpiWorld().broadcast(sourceIndex, toSend);
const int sendCount = 2;
std::vector<MyStruct> vecToSend(sendCount*mpiWorld().size());
std::vector<MyStruct> scattered = mpiWorld().scatter(sourceIndex, vecToSend, sendCount);
std::vector<MyStruct> gathered = mpiWorld().gather(sourceIndex,toSend);
std::vector<MyStruct> allGathered = mpiWorld().allGather(toSend);
```

Their varying counterparts are

```c++
std::vector<int> sendCounts(mpiWorld().size());
std::vector<MyStruct> scatteredv = mpiWorld().varyingScatter(sourceIndex,vecToSend,
	sendCounts); // Default displacements used
std::vector<int> displacements(mpiWorld().size());
std::vector<MyStruct> scatteredvTwo = mpiWorld().varyingScatter(sourceIndex,vecToSend,
	sendCounts,displacements);

std::vector<int> receiveCounts(mpiWorld().size());
std::vector<MyStruct> gatheredv = mpiWorld().varyingGather(sourceIndex,vecToSend,
	receiveCounts); // Default displacements used
std::vector<MyStruct> gatheredvTwo = mpiWorld().varyingGather(sourceIndex,vecToSend,
	receiveCounts,displacements);

std::vector<MyStruct> allGatheredv = mpiWorld().varyingAllGather(vecToSend,
	receiveCounts); // Default displacements used
std::vector<MyStruct> allGatheredvTwo = mpiWorld().varyingAllGather(vecToSend,
	receiveCounts,displacements);
```

# Communicator

## Identical v.s. Congruent communicators

When compared, MPI communicators can be identical or congruent. One can think of two identical MPI communicators as two pointers pointing on the same object. On the other hand, two congruent (but *not* identical) MPI communicators can be seen as two distinct objects that contain the same information (this analogy is a bit misleading since two congruent MPI communicators usually have different context, but it is enough to understand what is below).

The lines of codes

```c++
MPI_Comm mpiX;
Communicator x(mpiX);
```

create a `Communicator` that contains an MPI implementation which is **congruent** (and *not* identical) to `mpiX`. The MPI implementation for the communicator `x` has its own space in memory, and this memory is freed when `x` is destroyed. Of course, copy operations imply congruence, but not identity,

```c++
Communicator congruentToX(x);
```

It is possible to obtain `Communicator`s that have **identical** MPI implementation to `MPI_COMM_WORLD` and `MPI_COMM_SELF` by using the functions `mpiWolrd()` and `mpiSelf()`. Moreover, the function `createProxy(mpiX)` returns a `Communicator` that has an **identical** MPI implementation to `mpiX`. This property is preserved when the proxy is moved

```c++
Communicator proxy = createProxy(mpiX);
Communicator identicalToProxy(std::move(proxy));
```

However, of course, if you copy a proxy, a new MPI implementation is created, as always when a copy is made

```c++
Communicator congruentToProxy(identicalToProxy);
```

In case of doubt, you can always use the functions `areCongruent` and `areIdentical` to compare two communicators.

```c++
Communicator a;
Communicator b(a);
areCongruent(a,b); // true
areIdentical(a,b); // false
```

## Other Features

`Communicator`s can be splitted. For instance, to create two communicators, one that contains every processes with even rank and the other that contains every processes with odd rank, one can use

```c++
const int color = a.rank() % 2;
const int key = a.rank();
Communicator splitted = a.split(color,key);
```
The color select the `Communicator` in which the current process ends up, and the key determines the rank of this process in the new `Communicator`.

# Documentation

Documentation of this project can be built using [doxygen](http://www.doxygen.org).

# References

- [An excellent MPI tutorial](http://mpitutorial.com/tutorials/).

<a name="footnoteOne"><sup>1</sup></a> The library is tested with clang version 3.8.0-2ubuntu4 (tags/RELEASE_380/final) and gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4).
