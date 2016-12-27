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

#include <cassert>
#include <exception> // std::terminate
#include <stdexcept> // std::runtime_error
#include <string> // std::to_string
#include <type_traits> // std::is_pod, std::enable_if
#include <mpi.h>

/** \brief An alternative to Boost.MPI for a user friendly C++ interface for MPI (MPICH). **/
namespace NiceMPI {

/** \brief Forward declaration. */
class Communicator &mpiWorld();

/** \brief Initialize and finalize MPI using RAII. */
struct MPI_RAII {
	/** \brief Initializes MPI. */
	MPI_RAII(int argc, char* argv[]) {
		int error = MPI_Init(&argc, &argv);
		if(error != MPI_SUCCESS) throw std::runtime_error("Error code " + std::to_string(error) + " in MPI.");
	}
	/** \brief Finalizes MPI. */
	~MPI_RAII() {
		int error = MPI_Finalize();
		assert(error == MPI_SUCCESS); // guarantee stack unwinding in debug mode
		if(error != MPI_SUCCESS) std::terminate();
	}
};

/** \brief Represents a MPI communitator. */
class Communicator {
public:
	Communicator() {
		handleError(MPI_Comm_dup(MPI_COMM_WORLD, &mpiCommunicator));
	}
	Communicator(const Communicator& rhs) {
		handleError(MPI_Comm_dup(rhs.mpiCommunicator, &mpiCommunicator));
	}
	Communicator(Communicator&& rhs): Communicator() {
		std::swap(mpiCommunicator,rhs.mpiCommunicator);
	}
	~Communicator() {
		handleError(MPI_Comm_free(&mpiCommunicator));
	}
	Communicator& operator=(const Communicator& rhs) {
		if(this != &rhs) handleError(MPI_Comm_dup(rhs.mpiCommunicator, &mpiCommunicator));
		return *this;
	}
	Communicator& operator=(Communicator&& rhs) {
		std::swap(mpiCommunicator,rhs.mpiCommunicator);
		return *this;
	}

	/** \brief Returns the MPI communicator associated to \p this. This method breaks encapsulation, but it is
  provided in order to facilitate the interface with MPI functions not implemented here. Minimize its use. */
	MPI_Comm get() const {
		return mpiCommunicator;
	}
	int rank() const {
		int rank;
		handleError(MPI_Comm_rank(mpiCommunicator, &rank));
		return rank;
	}
	int size() const {
		int size;
		handleError(MPI_Comm_size(mpiCommunicator, &size));
		return size;
	}
	Communicator split(int color, int key) const {
		MPI_Comm splitted;
		handleError(MPI_Comm_split(mpiCommunicator,color,key,&splitted));
		return Communicator{std::move(splitted)};
	}

	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	Type broadcast(int source, Type data) {
		handleError(MPI_Bcast(&data,sizeof(data),MPI_UNSIGNED_CHAR,source,mpiCommunicator));
		return data;
	}
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> gather(int source, Type data) {
		std::vector<Type> result;
		if(rank() == source) result.resize(size());
		handleError(MPI_Gather(&data,sizeof(data),MPI_UNSIGNED_CHAR,result.data(),sizeof(data),MPI_UNSIGNED_CHAR,source,
			mpiCommunicator));
		return result;
	}
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> allGather(Type data) {
		std::vector<Type> result(size());
		handleError(MPI_Allgather(&data,sizeof(data),MPI_UNSIGNED_CHAR,result.data(),sizeof(data),MPI_UNSIGNED_CHAR,
			mpiCommunicator));
		return result;
	}
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	Type receiveAndBlock(int source, int tag = 0) {
		Type data;
		handleError(MPI_Recv(&data,sizeof(data),MPI_UNSIGNED_CHAR,source,tag,mpiCommunicator,MPI_STATUS_IGNORE));
		return data;
	}
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> scatter(int source, std::vector<Type> toSend, int sendCount) {
		assert(rank() != source or (static_cast<int>(toSend.size()) - sendCount*size()) >= 0 );
		std::vector<Type> result(sendCount);
		handleError(MPI_Scatter(toSend.data(), sendCount*sizeof(Type), MPI_UNSIGNED_CHAR, 
			result.data(), sendCount*sizeof(Type), MPI_UNSIGNED_CHAR, source, mpiCommunicator));
		return result;
	}
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	void sendAndBlock(Type data, int destination, int tag = 0) {
		handleError(MPI_Send(&data,sizeof(data),MPI_UNSIGNED_CHAR,destination,tag,mpiCommunicator));
	}

	friend Communicator &mpiWorld() {
		thread_local Communicator* neverDestructedWorld = new Communicator{MPI_Comm{MPI_COMM_WORLD}};
		return *neverDestructedWorld;
	}

private:
	Communicator(MPI_Comm&& mpiCommunicatorRhs) {
		mpiCommunicator = mpiCommunicatorRhs;
	}
	static void handleError(int error) {
		if(error != MPI_SUCCESS) throw std::runtime_error("Error code " + std::to_string(error) + " in MPI.");
	}

	MPI_Comm mpiCommunicator;
};


} // NiceMPi

#endif  /* NICEMPI_H */
