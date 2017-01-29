/* MIT License

Copyright (c) 2016 Kevin Lalumiere

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#ifndef NICEMPI_H
#define NICEMPI_H

#include <stdexcept> // std::runtime_error
#include <type_traits> // std::is_pod, std::enable_if
#include <mpi.h>

#define UNUSED(x) ((void)x)

/** \brief An alternative to Boost.MPI for a user friendly C++ interface for MPI (MPICH). **/
namespace NiceMPI {

/** \brief Forward declaration. */
class Communicator &mpiWorld();

class NiceMPIexception: public std::runtime_error {
public:
	NiceMPIexception(int error);

	const int error;
};

/** \brief Initialize and finalize MPI using RAII. */
struct MPI_RAII {
	/** \brief Initializes MPI. */
	MPI_RAII(int argc, char* argv[]);
	/** \brief Finalizes MPI. */
	~MPI_RAII();
};

/** \brief Represents a MPI communitator. */
class Communicator {
public:
	Communicator();
	Communicator(const Communicator& rhs);
	Communicator(Communicator&& rhs);
	~Communicator();
	Communicator& operator=(const Communicator& rhs);
	Communicator& operator=(Communicator&& rhs);

	/** \brief Returns the MPI communicator associated to \p this. This method breaks encapsulation, but it is
  provided in order to facilitate the interface with MPI functions not implemented here. Minimize its use. */
	MPI_Comm get() const;
	int rank() const;
	int size() const;
	Communicator split(int color, int key) const;


	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> allGather(Type data);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	Type broadcast(int source, Type data);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> gather(int source, Type data);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	Type receiveAndBlock(int source, int tag = 0);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> scatter(int source, const std::vector<Type>& toSend, int sendCount);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	void sendAndBlock(Type data, int destination, int tag = 0);

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingAllGather(const std::vector<Type>& data, const std::vector<int>& receiveCounts,
		const std::vector<int>& displacements = {});

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingGather(int source, const std::vector<Type>& data, const std::vector<int>& receiveCounts,
		const std::vector<int>& displacements = {});

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingScatter(int source, const std::vector<Type>& toSend, const std::vector<int>& sendCounts,
		const std::vector<int>& displacements = {});


	friend Communicator &mpiWorld() {
		thread_local Communicator* neverDestructedWorld = new Communicator{MPI_Comm{MPI_COMM_WORLD}};
		return *neverDestructedWorld;
	}

private:
	Communicator(MPI_Comm&& mpiCommunicatorRhs);
	static std::vector<int> createDefaultDisplacements(const std::vector<int>& sendCounts);
	template<typename Type>
	static std::vector<int> createScaledDisplacements(std::vector<int> displacements);
	static void handleError(int error);
	static int sum(const std::vector<int>& data);
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingScatterImpl(int source, const std::vector<Type>& toSend,const std::vector<int>& sendCounts,
		const std::vector<int>& scaledSendCounts, const std::vector<int>& displacements);

	MPI_Comm mpiCommunicator;
};


} // NiceMPi

#include "NiceMPI.hpp"

#endif  /* NICEMPI_H */
