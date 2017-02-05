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
#include <NiceMPI/NiceMPIexception.h> // handleError

namespace NiceMPI {

inline Communicator::Communicator(MPI_Comm mpiCommunicator): handle(mpiCommunicator)
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


template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline std::vector<Type> Communicator::allGather(Type data) {
	std::vector<Type> result(size());
	handleError(MPI_Allgather(&data,sizeof(Type),MPI_UNSIGNED_CHAR,result.data(),sizeof(Type),MPI_UNSIGNED_CHAR,
		handle.get() ));
	return result;
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
inline std::vector<typename Collection::value_type> Communicator::allGather(const Collection& data) {
	using Type = typename Collection::value_type;
	std::vector<Type> result(size()*data.size());
	handleError(MPI_Allgather(data.data(),sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR,result.data(),
		sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR, handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline ReceiveRequest<Type> Communicator::asyncReceive(int source, int tag) {
	ReceiveRequest<Type> r(1);
	handleError(MPI_Irecv(r.data.data(),sizeof(Type),MPI_UNSIGNED_CHAR,source,tag,handle.get(),&r.value));
	return r;
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
inline ReceiveRequest<Collection> Communicator::asyncReceive(int count, int source, int tag) {
	using Type = typename Collection::value_type;
	ReceiveRequest<Collection> r(count);
	handleError(MPI_Irecv(r.data.data(),sizeof(Type)*count,MPI_UNSIGNED_CHAR,source,tag,handle.get(),&r.value));
	return r;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline SendRequest Communicator::asyncSend(Type data, int destination, int tag) {
	MPI_Request x;
	handleError(MPI_Isend(&data,sizeof(Type),MPI_UNSIGNED_CHAR,destination,tag,handle.get(),&x));
	return SendRequest(x);
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
inline SendRequest Communicator::asyncSend(const Collection& data, int destination, int tag) {
	using Type = typename Collection::value_type;
	MPI_Request x;
	handleError(MPI_Isend(data.data(),sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR,destination,tag,handle.get(),&x));
	return SendRequest(x);
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline Type Communicator::broadcast(int source, Type data) {
	handleError(MPI_Bcast(&data,sizeof(Type),MPI_UNSIGNED_CHAR,source,handle.get() ));
	return data;
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
inline Collection Communicator::broadcast(int source, Collection data) {
	using Type = typename Collection::value_type;
	auto sizeToBroadcast = broadcast(source,data.size());
	if(rank() != source) data = initializeWithCount(Collection{},sizeToBroadcast);
	handleError(MPI_Bcast(data.data(),sizeof(Type)*sizeToBroadcast,MPI_UNSIGNED_CHAR,source,handle.get() ));
	return data;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline std::vector<Type> Communicator::gather(int source, Type data) {
	std::vector<Type> result;
	if(rank() == source) result.resize(size());
	handleError(MPI_Gather(&data,sizeof(Type),MPI_UNSIGNED_CHAR,result.data(),sizeof(Type),MPI_UNSIGNED_CHAR,source,
		handle.get() ));
	return result;
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
std::vector<typename Collection::value_type> Communicator::gather(int source, const Collection& data) {
	using Type = typename Collection::value_type;
	std::vector<Type> result;
	if(rank() == source) result.resize(size()*data.size());
	handleError(MPI_Gather(data.data(),sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR,result.data(),
		sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR,source,handle.get() ));
	return result;
}

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline Type Communicator::receiveAndBlock(int source, int tag) {
	Type data;
	handleError(MPI_Recv(&data,sizeof(Type),MPI_UNSIGNED_CHAR,source,tag,handle.get() ,MPI_STATUS_IGNORE));
	return data;
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
Collection Communicator::receiveAndBlock(int count, int source, int tag) {
	Collection data = initializeWithCount(Collection{},count);
	using Type = typename Collection::value_type;
	handleError(MPI_Recv(data.data(),sizeof(Type)*count,MPI_UNSIGNED_CHAR,source,tag,handle.get() ,MPI_STATUS_IGNORE));
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

template<typename Type, typename std::enable_if<std::is_pod<Type>::value and !is_std_array<Type>::value,bool>::type>
inline void Communicator::sendAndBlock(Type data, int destination, int tag) {
	handleError(MPI_Send(&data,sizeof(Type),MPI_UNSIGNED_CHAR,destination,tag,handle.get() ));
}

template<class Collection,
	typename std::enable_if<std::is_pod<typename Collection::value_type>::value,bool>::type
>
inline void Communicator::sendAndBlock(const Collection& data, int destination, int tag) {
	using Type = typename Collection::value_type;
	handleError(MPI_Send(data.data(),sizeof(Type)*data.size(),MPI_UNSIGNED_CHAR,destination,tag,handle.get() ));
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


inline Communicator::Communicator(MPI_Comm* mpiCommunicatorRhs): handle(mpiCommunicatorRhs)
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

template<typename Type>
inline std::vector<Type> Communicator::initializeWithCount(std::vector<Type>, int count) {
	return std::vector<Type>(count);
}
template<typename Type, std::size_t N>
inline std::array<Type,N> Communicator::initializeWithCount(std::array<Type,N> a, int /*count*/) {
	return a;
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



inline bool areCongruent(const Communicator& a, const Communicator& b) {
	int result;
	MPI_Comm_compare(a.get(), b.get(), &result);
	return result == MPI_CONGRUENT;
}
inline bool areIdentical(const Communicator& a, const Communicator& b) {
	int result;
	MPI_Comm_compare(a.get(), b.get(), &result);
	return result == MPI_IDENT;
}

} // NiceMPi

#endif  /* NICEMPI_HPP */
