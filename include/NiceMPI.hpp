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

#ifndef NICEMPI_HPP
#define NICEMPI_HPP

#include <cassert>
#include <exception> // std::terminate
#include <string> // std::to_string
#include <utility> // std::move

namespace NiceMPI {

inline NiceMPIexception::NiceMPIexception(int error)
: std::runtime_error("Error code " + std::to_string(error) + " in MPI."), error(error)
{}



inline MPI_RAII::MPI_RAII(int argc, char* argv[]) {
	int error = MPI_Init(&argc, &argv);
	if(error != MPI_SUCCESS) throw NiceMPIexception{error};
}
inline MPI_RAII::~MPI_RAII() {
	int error = MPI_Finalize();
	if(error != MPI_SUCCESS) std::terminate();
}

/** \brief Implements a MPI communicator handle that owns the MPI_Comm. */
class OwnedCommunicator: public MPIcommunicatorHandleImpl {
public:
	/** \brief Creates a communicator handle congruent (but not equal) to rhs. */
	OwnedCommunicator(MPI_Comm rhs) {
		handleError(MPI_Comm_dup(rhs,&mpiCommunicator));
	}
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator(const OwnedCommunicator&) = delete;
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator(OwnedCommunicator&&) = delete;
	/** \brief Destroys the underlying MPI communicator. */
	~OwnedCommunicator() {
		int error = MPI_Comm_free(&mpiCommunicator);
		if(error != MPI_SUCCESS) std::terminate();
	}
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator& operator=(const OwnedCommunicator&) = delete;
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator& operator=(OwnedCommunicator&&) = delete;
	std::unique_ptr<MPIcommunicatorHandleImpl> clone() const override {
		return std::unique_ptr<MPIcommunicatorHandleImpl>(new OwnedCommunicator(mpiCommunicator));
	}
	MPI_Comm get() const override {
		return mpiCommunicator;
	}
	MPI_Comm get() override {
		return mpiCommunicator;
	}

private:
	/** \brief Underlying MPI implementation of this handle. */
	MPI_Comm mpiCommunicator;	
};

/** \brief Implements a MPI communicator handle that is only a proxy for the MPI_Comm. */
class ProxyCommunicator: public MPIcommunicatorHandleImpl {
public:
	/** \brief Creates a communicator handle equal to rhs. */
	ProxyCommunicator(MPI_Comm rhs): mpiCommunicator(rhs)
	{}
	std::unique_ptr<MPIcommunicatorHandleImpl> clone() const override {
		return std::unique_ptr<MPIcommunicatorHandleImpl>(new OwnedCommunicator(mpiCommunicator)); // Can't copy proxy
	}
	MPI_Comm get() const override {
		return mpiCommunicator;
	}
	MPI_Comm get() override {
		return mpiCommunicator;
	}

private:
	/** \brief Underlying MPI implementation of this handle. */
	MPI_Comm mpiCommunicator;
};

inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPI_Comm mpiCommunicator)
: impl(new OwnedCommunicator(mpiCommunicator))
{}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPI_Comm* mpiCommunicator)
: impl( new ProxyCommunicator(*mpiCommunicator) )
{}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(const MPIcommunicatorHandle& rhs) {
	impl = rhs.impl->clone();
}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPIcommunicatorHandle&& rhs) {
	impl = std::move(rhs.impl);
}
inline MPIcommunicatorHandle::~MPIcommunicatorHandle() = default;
inline MPIcommunicatorHandle& MPIcommunicatorHandle::operator=(const MPIcommunicatorHandle& rhs) {
	impl = rhs.impl->clone();
	return *this;
}
inline MPIcommunicatorHandle& MPIcommunicatorHandle::operator=(MPIcommunicatorHandle&& rhs) {
	impl = std::move(rhs.impl);
	return *this;
}
inline MPI_Comm MPIcommunicatorHandle::get() const {
	return impl->get();
}
inline MPI_Comm MPIcommunicatorHandle::get() {
	return impl->get();
}



inline Communicator::Communicator(): handle(MPI_COMM_WORLD)
{}

inline MPI_Comm Communicator::get() const {
	return handle.get() ;
}

inline int Communicator::rank() const {
	int rank;
	handleError(MPI_Comm_rank(handle.get() , &rank));
	return rank;
}

inline int Communicator::size() const {
	int size;
	handleError(MPI_Comm_size(handle.get() , &size));
	return size;
}

inline Communicator Communicator::split(int color, int key) const {
	MPI_Comm splitted;
	handleError(MPI_Comm_split(handle.get() ,color,key,&splitted));
	return Communicator{&splitted};
}


template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::allGather(Type data) {
	std::vector<Type> result(size());
	handleError(MPI_Allgather(&data,sizeof(Type),MPI_UNSIGNED_CHAR,result.data(),sizeof(Type),MPI_UNSIGNED_CHAR,
		handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline Type Communicator::broadcast(int source, Type data) {
	handleError(MPI_Bcast(&data,sizeof(Type),MPI_UNSIGNED_CHAR,source,handle.get() ));
	return data;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::gather(int source, Type data) {
	std::vector<Type> result;
	if(rank() == source) result.resize(size());
	handleError(MPI_Gather(&data,sizeof(Type),MPI_UNSIGNED_CHAR,result.data(),sizeof(Type),MPI_UNSIGNED_CHAR,source,
		handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline Type Communicator::receiveAndBlock(int source, int tag) {
	Type data;
	handleError(MPI_Recv(&data,sizeof(Type),MPI_UNSIGNED_CHAR,source,tag,handle.get() ,MPI_STATUS_IGNORE));
	return data;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::scatter(int source, const std::vector<Type>& toSend, int sendCount) {
	const bool enoughDataToSend = (static_cast<int>(toSend.size()) - sendCount*size()) >= 0;
	assert(rank() != source or enoughDataToSend); UNUSED(enoughDataToSend);
	std::vector<Type> result(sendCount);
	handleError(MPI_Scatter(toSend.data(), sendCount*sizeof(Type), MPI_UNSIGNED_CHAR, 
		result.data(), sendCount*sizeof(Type), MPI_UNSIGNED_CHAR, source, handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline void Communicator::sendAndBlock(Type data, int destination, int tag) {
	handleError(MPI_Send(&data,sizeof(Type),MPI_UNSIGNED_CHAR,destination,tag,handle.get() ));
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::varyingAllGather(const std::vector<Type>& data,
	const std::vector<int>& receiveCounts, const std::vector<int>& displacements)
{
	std::vector<Type> result(sum(receiveCounts));
	std::vector<int> scaledReceiveCounts(receiveCounts);
	for(auto&& x: scaledReceiveCounts) x *= sizeof(Type);
	const std::vector<int> scaledDisplacements = displacements.empty() ?
		createDefaultDisplacements(scaledReceiveCounts) :
		createScaledDisplacements<Type>(displacements);

	handleError(MPI_Allgatherv(data.data(), data.size()*sizeof(Type), MPI_UNSIGNED_CHAR, result.data(),
		scaledReceiveCounts.data(), scaledDisplacements.data(), MPI_UNSIGNED_CHAR, handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::varyingGather(int source, const std::vector<Type>& data,
	const std::vector<int>& receiveCounts, const std::vector<int>& displacements)
{
	std::vector<Type> result;
	std::vector<int> scaledReceiveCounts;
	std::vector<int> scaledDisplacements;
	if(rank() == source) {
		result.resize(sum(receiveCounts));
		scaledReceiveCounts = receiveCounts;
		for(auto&& x: scaledReceiveCounts) x *= sizeof(Type);
		if(displacements.empty()) scaledDisplacements = createDefaultDisplacements(scaledReceiveCounts);
		else scaledDisplacements = createScaledDisplacements<Type>(displacements);
	}
	handleError(MPI_Gatherv(data.data(), data.size()*sizeof(Type), MPI_UNSIGNED_CHAR, result.data(),
		scaledReceiveCounts.data(), scaledDisplacements.data(), MPI_UNSIGNED_CHAR, source, handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::varyingScatter(int source, const std::vector<Type>& toSend,
	const std::vector<int>& sendCounts, const std::vector<int>& displacements)
{
	const auto enoughDataToSend = [&] () {
		decltype(toSend.size()) sumOfSendCounts = 0;
		for(auto&& x: sendCounts) {
			assert(x >= 0);
			sumOfSendCounts += x;
		};
		return toSend.size() >= sumOfSendCounts;
	};
	assert(rank() != source or enoughDataToSend()); UNUSED(enoughDataToSend);
	assert(static_cast<int>(sendCounts.size()) >= size());

	std::vector<int> scaledSendCounts(sendCounts);
	for(auto&& x: scaledSendCounts) x *= sizeof(Type);
	const std::vector<int> scaledDisplacements = displacements.empty() ?
		createDefaultDisplacements(scaledSendCounts) :
		createScaledDisplacements<Type>(displacements);
	return varyingScatterImpl(source,toSend,sendCounts,scaledSendCounts,scaledDisplacements);
}


inline Communicator::Communicator(MPI_Comm* rhs): handle(rhs)
{}

inline std::vector<int> Communicator::createDefaultDisplacements(const std::vector<int>& sendCounts) {
	std::vector<int> displacements(sendCounts.size());
	for(unsigned i = 1; i<sendCounts.size(); ++i) displacements[i] = displacements[i-1] + sendCounts[i-1];
	return displacements;
}

template<typename Type>
inline std::vector<int> Communicator::createScaledDisplacements(std::vector<int> displacements) {
	for(auto&& x: displacements) x *= sizeof(Type);
	return displacements;
}

inline int Communicator::sum(const std::vector<int>& data) {
	int theSum = 0;
	for(auto&& x: data) theSum += x;
	return theSum;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value,bool>::type>
inline std::vector<Type> Communicator::varyingScatterImpl(int source, const std::vector<Type>& toSend,
	const std::vector<int>& sendCounts, const std::vector<int>& scaledSendCounts,
	const std::vector<int>& displacements)
{
	std::vector<Type> result(sendCounts[rank()]);
	handleError(MPI_Scatterv(toSend.data(), scaledSendCounts.data(), displacements.data(), MPI_UNSIGNED_CHAR, 
		result.data(), scaledSendCounts[rank()], MPI_UNSIGNED_CHAR, source, handle.get() ));
	return result;
}



inline void handleError(int error) {
	if(error != MPI_SUCCESS) throw NiceMPIexception{error};
}

} // NiceMPi

#endif  /* NICEMPI_HPP */
