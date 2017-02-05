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

#include <array>
#include <cstddef> // std::size_t
#include <memory> // unique_ptr
#include <type_traits> // std::is_pod, std::enable_if
#include <utility> // std::move
#include <vector>
#include <mpi.h> // MPI_Comm
#include <NiceMPI/MPI_RAII.h> // for convenience
#include <NiceMPI/NiceMPIexception.h> // for convenience
#include "private/MPIcommunicatorHandle.h"

#define UNUSED(x) ((void)x)


/** \brief An alternative to Boost.MPI for a user friendly C++ interface for MPI (MPICH). **/
namespace NiceMPI {

class Communicator; // Forward declaration
Communicator &mpiWorld(); // Forward declaration
Communicator &mpiSelf(); // Forward declaration
Communicator createProxy(MPI_Comm mpiCommunicator); // Forward declaration



/** \brief std::array that contains PODs are both POD and collection, so we need to distinguish them. */
template<class T>
struct is_std_array {
	static constexpr bool value = false;
};
/** \brief std::array that contains PODs are both POD and collection, so we need to distinguish them.
  *Specialization*.*/
template<class T, std::size_t N>
struct is_std_array<std::array<T,N>> {
	static constexpr bool value = true;
};



/** \brief Returns after an asyncSend call, this object allows to control the status of the call. */
class SendRequest {
public:
	/** \brief Initializes this request with its MPI implementation. */
	SendRequest(MPI_Request value): value(value)
	{}
	/** \brief Returns true if the send operation is completed. */
	bool isCompleted() {
		int flag = 0;
		handleError(MPI_Test(&value, &flag, MPI_STATUS_IGNORE));
		return flag != 0;
	}
	/** \brief Waits for the data to be sent. */
	void wait() {
		handleError(MPI_Wait(&value,MPI_STATUS_IGNORE));
	}

private:
	/** \brief MPI implementation. */
	MPI_Request value;
};



/** \brief Returns after an asyncReceive call, this object allows to control the status of the call. */
template<class Type>
class ReceiveRequest {
public:
	/** \brief The function asyncReceive initializes the member of the request directly. */
	ReceiveRequest(): dataPtr(new Type)
	{}
	/** \brief Destroys the handle \p rhs. Only there because of the
		[rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	~ReceiveRequest() = default;
	/** \brief This object can only be moved, since the call to asyncReceive depends on the address of \p data. **/
	ReceiveRequest(const ReceiveRequest&) = delete;
	/** \brief This object can only be moved, since the call to asyncReceive depends on the address of \p data. **/
	ReceiveRequest(ReceiveRequest&&) = default;
	/** \brief This object can only be moved, since the call to asyncReceive depends on the address of \p data. **/
	ReceiveRequest& operator=(const ReceiveRequest&) = delete;
	/** \brief This object can only be moved, since the call to asyncReceive depends on the address of \p data. **/
	ReceiveRequest& operator=(ReceiveRequest&&) = delete;

	/** \brief Returns true if the receive operation is completed. */
	bool isCompleted() {
		int flag = 0;
		handleError(MPI_Test(&value, &flag, MPI_STATUS_IGNORE));
		return flag != 0;
	}
	/** \brief Waits for the data to be received. */
	void wait() {
		handleError(MPI_Wait(&value,MPI_STATUS_IGNORE));
	}
	/** Returns the data, assuming that the user made sure that the receiving operation was completed. */
	std::unique_ptr<Type> take() {
		return std::move(dataPtr);
	}

	/** \brief The function asyncReceive needs the address of \p data. */
	friend Communicator;

private:
	/** \brief MPI implementation. */
	MPI_Request value;
	/** \brief \p data to be received. */
	std::unique_ptr<Type> dataPtr;
};



/** \brief Represents a MPI communitator. */
class Communicator {
public:
	/** \brief Creates a communicator congruent (but not equal) to MPI_COMM_WORLD. */
	explicit Communicator(MPI_Comm mpiCommunicator = MPI_COMM_WORLD);
	/** \brief Returns the MPI communicator associated to \p this. This method breaks encapsulation, but it is
  provided in order to facilitate the interface with MPI functions not implemented here. Minimize its use. */
	MPI_Comm get() const;
	/** \brief Returns the rank of the process in this communicator. */
	int rank() const;
	/** \brief Returns the size of this communicator. */
	int size() const;
	/** \brief Splits this communicator. 'Processes with the same \p color are in the same new communicator.' The \p
  key control of rank assignment.*/
	Communicator split(int color, int key) const;


	/** \brief Regroups the \p data of every processes in a single vector and returns it. */
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> allGather(Type data);

	/** \brief Starts to receive data of type \p Type from the \p source. A \p tag can be required to be provided
  with the data. \p MPI_ANY_TAG can be used. Returns a ReceiveRequest object that can be used to find out if
  the data were received, or to wait until they are received and get them.*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	ReceiveRequest<Type> asyncReceive(int source, int tag = 0);

	/** \brief Starts to send \p data to the \p destination. A \p tag can be required to be provided with the data.
  \p MPI_ANY_TAG can be used. Returns a SendRequest object that can be used to find out if the data were sent, or
  to wait until they are sent.*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	SendRequest asyncSend(Type data, int destination, int tag = 0);

	/** \brief The \p source broadcast its \p data to every processes. */
	template<typename Type,
		typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type = true
	>
	Type broadcast(int source, Type data);

	/** \brief The \p source broadcast its \p data to every processes. */
	template<class Collection,
		typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type = true
	>
	Collection broadcast(int source, Collection data);

	/** \brief The \p source gathers the \p data of every processes. */
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> gather(int source, Type data);

	/** \brief Wait to receive data of type \p Type from the \p source. A \p tag can be required to be provided with
  the data. \p MPI_ANY_TAG can be used.*/
	template<typename Type,
		typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type = true
	>
	Type receiveAndBlock(int source, int tag = 0);

	/** \brief Wait to receive data of type \p Type from the \p source. A \p tag can be required to be provided with
  the data. \p MPI_ANY_TAG can be used.*/
	template<class Collection,
		typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type = true
	>
	Collection receiveAndBlock(int count, int source, int tag = 0);

	/** \brief The \p source scatters \p sendCount of its data \p toSend to every processes. Hence, the process with
  rank \p i receives the data from \p toSend[i] to toSend[i+\p sendCount].*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> scatter(int source, const std::vector<Type>& toSend, int sendCount);

	/** \brief Wait to send \p data to the \p destination. A \p tag can be required to be provided with
  the data. \p MPI_ANY_TAG can be used.*/
	template<typename Type,
		typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type = true
	>
	void sendAndBlock(Type data, int destination, int tag = 0);

	/** \brief Wait to send \p data to the \p destination. A \p tag can be required to be provided with
  the data. \p MPI_ANY_TAG can be used.*/
	template<class Collection,
		typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type = true
	>
	void sendAndBlock(const Collection& data, int destination, int tag = 0);

	/** \brief Regroups the \p data of every processes in a single vector and returns it. \p receiveCounts[i] data
  is received from the process with rank \p i. These data starts at the index \p displacements[i] of the
  returned vector.*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingAllGather(const std::vector<Type>& data, const std::vector<int>& receiveCounts,
		const std::vector<int>& displacements = {});

	/** \brief The \p source gathers the \p data of every processes. \p receiveCounts[i] data
  is received from the process with rank \p i. These data starts at the index \p displacements[i] of the
  returned vector. If the argument \p displacements is empty, the data are placed sequentially in the returned vector.*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingGather(int source, const std::vector<Type>& data, const std::vector<int>& receiveCounts,
		const std::vector<int>& displacements = {});

	/** \brief The \p source scatters the data \p toSend of every processes. \p sendCounts[i] data is sent to the
  process with rank \p i. These data are taken starting from the index \p displacements[i] of the vector \p
  toSend. If the argument \p displacements is empty, the data are taken sequentially in the vector \p toSend.*/
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingScatter(int source, const std::vector<Type>& toSend, const std::vector<int>& sendCounts,
		const std::vector<int>& displacements = {});


	/** \brief Returns a proxy communicator identical to \p mpiCommunicator. */
	friend Communicator createProxy(MPI_Comm mpiCommunicator) {
		return Communicator{&mpiCommunicator};
	}
	/** \brief Returns a communicator identical to \p MPI_COMM_WORLD. */
	friend Communicator &mpiWorld() {
		thread_local MPI_Comm notRvalue = MPI_COMM_WORLD;
		thread_local Communicator proxy(&notRvalue);
		return proxy;
	}
	/** \brief Returns a communicator identical to \p MPI_COMM_SELF. */
	friend Communicator &mpiSelf() {
		thread_local MPI_Comm notRvalue = MPI_COMM_SELF;
		thread_local Communicator proxy(&notRvalue);
		return proxy;
	}

private:
	/** \brief Creates a proxy communicator identical to \p mpiCommunicatorRhs. */
	Communicator(MPI_Comm* mpiCommunicatorRhs);
	/** \brief Returns a displacement vector that corresponds to the \p sendCounts[i] data placed sequentially. */
	static std::vector<int> createDefaultDisplacements(const std::vector<int>& sendCounts);
	/** \brief Returns the \p displacements scaled by the size of \p Type.*/
	template<typename Type>
	static std::vector<int> createScaledDisplacements(std::vector<int> displacements);
	/** \brief Initializes the collection with \p count elements. */
	template<typename Type>
	static std::vector<Type> initializeWithCount(std::vector<Type>, int count);
	/** \brief Initializes the collection with \p count elements. */
	template<typename Type, std::size_t N>
	static std::array<Type,N> initializeWithCount(std::array<Type,N> a, int /*count*/);
	/** \brief Returns the sum of the \p data. */
	static int sum(const std::vector<int>& data);
	/** \brief Implements \p varyingScatter(). */
	template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type = true>
	std::vector<Type> varyingScatterImpl(int source, const std::vector<Type>& toSend,const std::vector<int>& sendCounts,
		const std::vector<int>& scaledSendCounts, const std::vector<int>& displacements);

	/** \brief Handles the life of the MPI implementation of \p this communicator. */
	MPIcommunicatorHandle handle;
};

/** \brief Returns true if the communicators are congruent. */
bool areCongruent(const Communicator& a, const Communicator& b);
/** \brief Returns true if the communicators are identical. */
bool areIdentical(const Communicator& a, const Communicator& b);

} // NiceMPi

#include "private/NiceMPI.hpp"

#endif  /* NICEMPI_H */
