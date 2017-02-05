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

#include <chrono> // std::chrono::microseconds
#include <thread> // std::this_thread::sleep_for;
#include <utility> // std::move
#include <vector>
#include <gtest/gtest.h>
#include <NiceMPI/NiceMPI.h>
#include "buildInformationNiceMPI.h"

using namespace NiceMPI;

class NiceMPItests : public ::testing::Test {
public:
	struct PODtype {
		int theInt;
		double theDouble;
		char theChar;
	};

	bool areCongruentMPI(const MPI_Comm &a, const MPI_Comm &b) const {
		int result;
		MPI_Comm_compare(a, b, &result);
		return result == MPI_CONGRUENT;
	}
	std::vector<PODtype> createPODtypeCollection(const int count) {
		if(mpiWorld().rank() != sourceIndex) return {};
		std::vector<PODtype> result;
		for(int i = 0; i < count; ++i) result.push_back(createPODtypeForRank(i));
		return result;
	}
	PODtype createPODtypeForRank(int rank) {
		PODtype result{podTypeInstance};
		result.theInt = rank*2;
		return result;
	}
	void expectGathered(const std::vector<PODtype>& gathered) {
		ASSERT_EQ(mpiWorld().size(),gathered.size());
		for(int i=0; i<mpiWorld().size(); ++i) {
			PODtype expected{podTypeInstance};
			expected.theInt = 2*i;
			expectNear(expected,gathered[i],defaultTolerance);
		}
	}
	void expectGatheredCollection(const std::vector<PODtype>& gathered, const int sizeByProcess) {
		ASSERT_EQ(mpiWorld().size()*sizeByProcess,gathered.size());
		for(int i=0; i<mpiWorld().size(); ++i) {
			PODtype expected{podTypeInstance};
			expected.theInt = 2*i;
			expectNear(expected,gathered[sizeByProcess*i],defaultTolerance);
			expectNear(expected,gathered[sizeByProcess*i+1],defaultTolerance);
		}
	}
	void expectNear(const PODtype& expected, const PODtype& actual, double tolerance) {
		EXPECT_EQ(expected.theInt, actual.theInt);
		EXPECT_NEAR(expected.theDouble, actual.theDouble, tolerance);
		EXPECT_EQ(expected.theChar, actual.theChar);
	}
	template<class CollectionType>
	void testAsyncSendAndReceiveCollection() {
		if(sourceIndex == destinationIndex) return;
		CollectionType toSend;
		const int count = 2;
		if(mpiWorld().rank() == sourceIndex) {
			toSend = {{ podTypeInstance, podTypeInstance }};
			SendRequest r = mpiWorld().asyncSend(toSend,destinationIndex);
			r.wait();
		}
		if(mpiWorld().rank() == destinationIndex) {
			ReceiveRequest<CollectionType> r = mpiWorld().asyncReceive<CollectionType>(count,sourceIndex);
			r.wait();
			std::vector<PODtype> data = r.take();
			EXPECT_EQ(count,data.size());
			for(auto&& x: data) expectNear(podTypeInstance, x, defaultTolerance);
		}
	}
	template<class CollectionType>
	void testBroadcastCollection() {
		CollectionType data;
		if(mpiWorld().rank() == sourceIndex) data = {{ podTypeInstance, podTypeInstance }};
		const CollectionType results = mpiWorld().broadcast(sourceIndex, data);
		EXPECT_EQ(2,results.size());
		for(auto&& x: results) expectNear(podTypeInstance, x, defaultTolerance);
	}
	template<class CollectionType>
	void testAllGather() {
		PODtype x = createPODtypeForRank(mpiWorld().rank());
		const int sizeByProcess = 2;
		const CollectionType toSend = {{ x, x }};
		const std::vector<PODtype> gathered = mpiWorld().allGather(toSend);
		expectGatheredCollection(gathered,sizeByProcess);
	}
	template<class CollectionType>
	void testGather() {
		PODtype x = createPODtypeForRank(mpiWorld().rank());
		const int sizeByProcess = 2;
		const CollectionType toSend = {{ x, x }};
		const std::vector<PODtype> gathered = mpiWorld().gather(sourceIndex, toSend );
		if(mpiWorld().rank()==sourceIndex) expectGatheredCollection(gathered,sizeByProcess);
		else EXPECT_EQ(0,gathered.size());
	}
	template<class CollectionType>
	void testSendAndReceiveCollection() {
		if(sourceIndex == destinationIndex) return;
		if(mpiWorld().rank() == sourceIndex) {
			const CollectionType toSend = {{ podTypeInstance, podTypeInstance }};
			mpiWorld().sendAndBlock(toSend,destinationIndex);
		}
		if(mpiWorld().rank() == destinationIndex) {
			const int count = 2;
			const CollectionType results = mpiWorld().receiveAndBlock<CollectionType>(count,sourceIndex);
			EXPECT_EQ(count,results.size());
			for(auto&& x: results) expectNear(podTypeInstance, x, defaultTolerance);
		}
	}

	const Communicator world;
	const int sourceIndex = 0;
	const int destinationIndex = world.size() -1;
	const double defaultTolerance = 1e-10;
	const PODtype podTypeInstance{42,6.66,'K'};
	const std::vector<PODtype> defaultCollection = createPODtypeCollection(mpiWorld().size());
};



TEST_F(NiceMPItests, areIdentical) {
	EXPECT_TRUE(areIdentical(world,world));
	const Communicator copy(world);
	EXPECT_FALSE(areIdentical(world,copy));
}
TEST_F(NiceMPItests, areCongruent) {
	EXPECT_FALSE(areCongruent(world,world));
	const Communicator copy(world);
	EXPECT_TRUE(areCongruent(world,copy));
}
TEST_F(NiceMPItests, defaultConstructorIsWorldCongruent) {
	EXPECT_TRUE( areCongruentMPI(MPI_COMM_WORLD,world.get()) );
}
TEST_F(NiceMPItests, mpiCommConstructor) {
	Communicator x(MPI_COMM_SELF);
	EXPECT_TRUE( areCongruentMPI(MPI_COMM_SELF, x.get()) );
}
TEST_F(NiceMPItests, rank) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	EXPECT_EQ(rank,world.rank());
}
TEST_F(NiceMPItests, size) {
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	EXPECT_EQ(size,world.size());
}
TEST_F(NiceMPItests, split) {
	const int color = world.rank() % 2;
	const int key = world.rank();
	Communicator splitted = world.split(color,key);

	int expectedSize = 0;
	if(maxWorldSize % 2 == 0) expectedSize = std::max(world.size()/2,1);
	else if(world.rank() % 2 == 0) expectedSize = world.size()/2 + 1;
	else expectedSize = world.size()/2;

	EXPECT_EQ(expectedSize,splitted.size());
	EXPECT_EQ(world.rank()/2,splitted.rank());
}


TEST_F(NiceMPItests, createProxy) {
	EXPECT_TRUE(areIdentical(world, createProxy(world.get()) ));
}
TEST_F(NiceMPItests, createProxyStored) {
	const Communicator proxy = createProxy(world.get());
	EXPECT_TRUE( areIdentical(world,proxy) );
}
TEST_F(NiceMPItests, copiedProxiesAreNotProxies) {
	const Communicator proxy = createProxy(world.get());
	const Communicator copy(proxy);
	EXPECT_TRUE( areCongruent(world,copy) );
}
TEST_F(NiceMPItests, movedProxiesAreProxies) {
	Communicator proxy = createProxy(world.get());
	const Communicator moved(std::move(proxy));
	EXPECT_TRUE( areIdentical(world,moved) );
}
TEST_F(NiceMPItests, mpiWorld) {
	EXPECT_TRUE( areIdentical(createProxy(MPI_COMM_WORLD),mpiWorld()) );
	EXPECT_TRUE( areIdentical(mpiWorld(),mpiWorld()) );
	EXPECT_TRUE( areCongruent(world,mpiWorld()) );
}
TEST_F(NiceMPItests, mpiSelf) {
	EXPECT_TRUE( areIdentical(createProxy(MPI_COMM_SELF),mpiSelf()) );
	EXPECT_TRUE( areIdentical(mpiSelf(),mpiSelf()) );
}


TEST_F(NiceMPItests, sendAndReceive) {
	if(sourceIndex == destinationIndex) return;
	const unsigned char toSend = 'K';
	if(mpiWorld().rank() == sourceIndex) mpiWorld().sendAndBlock(toSend,destinationIndex);
	if(mpiWorld().rank() == destinationIndex) {
		EXPECT_EQ(toSend,mpiWorld().receiveAndBlock<unsigned char>(sourceIndex));
	}
}
TEST_F(NiceMPItests, sendAndReceiveAnything) {
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) mpiWorld().sendAndBlock(podTypeInstance,destinationIndex);
	if(mpiWorld().rank() == destinationIndex) {
		expectNear(podTypeInstance, mpiWorld().receiveAndBlock<PODtype>(sourceIndex), defaultTolerance);
	}
}
TEST_F(NiceMPItests, sendAndReceiveWithTag) {
	if(sourceIndex == destinationIndex) return;
	const unsigned char toSend = 'K';
	const int tag = 3;
	if(mpiWorld().rank() == sourceIndex) mpiWorld().sendAndBlock(toSend,destinationIndex,tag);
	if(mpiWorld().rank() == destinationIndex) {
		EXPECT_EQ(toSend,mpiWorld().receiveAndBlock<unsigned char>(sourceIndex,MPI_ANY_TAG));
	}
}
TEST_F(NiceMPItests, broadcast) {
	PODtype data;
	if(mpiWorld().rank() == sourceIndex) data = podTypeInstance;
	expectNear(podTypeInstance, mpiWorld().broadcast(sourceIndex, data), defaultTolerance);
}
TEST_F(NiceMPItests, scatterOnlyOne) {
	const int sendCount = 1;
	const std::vector<PODtype> scattered = mpiWorld().scatter(sourceIndex,defaultCollection,sendCount);
	ASSERT_EQ(sendCount,scattered.size());
	expectNear(createPODtypeForRank(mpiWorld().rank()), scattered.at(0), defaultTolerance);
}
TEST_F(NiceMPItests, scatterTwoWithSpare) {
	const int sendCount = 2;
	const std::vector<PODtype> data = createPODtypeCollection((sendCount+1)*mpiWorld().size());
	const std::vector<PODtype> scattered = mpiWorld().scatter(sourceIndex,data,sendCount);
	ASSERT_EQ(sendCount,scattered.size());
	for(auto&& i: {0,1}) {
		expectNear(createPODtypeForRank(sendCount*mpiWorld().rank()+i), scattered.at(i), defaultTolerance);
	}
}
TEST_F(NiceMPItests, gather) {
	const std::vector<PODtype> gathered = mpiWorld().gather(sourceIndex, createPODtypeForRank(mpiWorld().rank()) );
	if(mpiWorld().rank()==sourceIndex) expectGathered(gathered);
	else EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, allGather) {
	const PODtype myData = createPODtypeForRank(mpiWorld().rank());
	expectGathered(mpiWorld().allGather(myData));
}


TEST_F(NiceMPItests, varyingScatterNothingSent) {
	const std::vector<int> sendCounts(mpiWorld().size());
	const std::vector<PODtype> scattered = mpiWorld().varyingScatter(sourceIndex,defaultCollection,sendCounts);
	EXPECT_EQ(0,scattered.size());
}
TEST_F(NiceMPItests, varyingScatterOneEach) {
	const std::vector<int> sendCounts(mpiWorld().size(),1);
	const std::vector<PODtype> scattered = mpiWorld().varyingScatter(sourceIndex,defaultCollection,sendCounts);
	ASSERT_EQ(sendCounts.at(0),scattered.size());
	expectNear(createPODtypeForRank(mpiWorld().rank()), scattered.at(0), defaultTolerance);
}
TEST_F(NiceMPItests, varyingScatterWithDisplacements) {
	std::vector<int> sendCounts(mpiWorld().size());
	sendCounts[destinationIndex] = std::max(1,mpiWorld().size()/2);
	std::vector<int> displacements(mpiWorld().size());
	displacements[destinationIndex] = mpiWorld().size()/2;

	const std::vector<PODtype> scattered = mpiWorld().varyingScatter(sourceIndex,defaultCollection,sendCounts,
		displacements);
	if(mpiWorld().rank() == destinationIndex) {
		ASSERT_EQ(sendCounts[destinationIndex],scattered.size());
		for(unsigned i = 0; i < scattered.size(); ++i) {
			expectNear(createPODtypeForRank(displacements[destinationIndex] + i), scattered[i], defaultTolerance);
		}
	}
	else {
		EXPECT_EQ(0,scattered.size());
	}
}
TEST_F(NiceMPItests, varyingGatherNothingGathered) {
	const std::vector<PODtype> data;
	const std::vector<int> receiveCounts(mpiWorld().size());
	const std::vector<PODtype> gathered = mpiWorld().varyingGather(sourceIndex,data,receiveCounts);
	EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, varyingGatherOneFromEach) {
	const std::vector<PODtype> data = { createPODtypeForRank(mpiWorld().rank()) };
	const std::vector<int> receiveCounts(mpiWorld().size(),1);
	const std::vector<PODtype> gathered = mpiWorld().varyingGather(sourceIndex,data,receiveCounts);
	if(mpiWorld().rank()==sourceIndex) expectGathered(gathered);
	else EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, varyingGatherWithDisplacements) {
	std::vector<PODtype> data;
	std::vector<int> receiveCounts(mpiWorld().size());
	std::vector<int> displacements(mpiWorld().size());
	if(mpiWorld().rank()==sourceIndex or mpiWorld().rank() == destinationIndex) {
		for(auto&&x : {mpiWorld().rank(), mpiWorld().rank() + 1}) data.emplace_back(createPODtypeForRank(x));
	}
	if(mpiWorld().rank()==sourceIndex) {
		receiveCounts[sourceIndex] = data.size();
		receiveCounts[destinationIndex] = data.size();
		displacements[sourceIndex] = data.size();
		displacements[destinationIndex] = 0;
	}
	const std::vector<PODtype> gathered = mpiWorld().varyingGather(sourceIndex,data,receiveCounts,displacements);
	if(mpiWorld().rank()==sourceIndex) {
		const std::vector<int> expectedOrder = { destinationIndex, destinationIndex +1, sourceIndex, sourceIndex + 1};
		if(mpiWorld().size() == 1) ASSERT_EQ(data.size(),gathered.size());
		else ASSERT_EQ(expectedOrder.size(),gathered.size());
		for(unsigned i = 0; i < gathered.size(); ++i) {
			expectNear(createPODtypeForRank(expectedOrder[i]), gathered[i], defaultTolerance);
		}
	}
	else EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, varyingAllGatherOneFromEach) {
	const std::vector<PODtype> data = { createPODtypeForRank(mpiWorld().rank()) };
	const std::vector<int> receiveCounts(mpiWorld().size(),1);
	const std::vector<PODtype> gathered = mpiWorld().varyingAllGather(data,receiveCounts);
	expectGathered(gathered);
}
TEST_F(NiceMPItests, varyingAllGatherWithDisplacements) {
	std::vector<PODtype> data;
	std::vector<int> receiveCounts(mpiWorld().size());
	std::vector<int> displacements(mpiWorld().size());

	if(mpiWorld().rank()==sourceIndex or mpiWorld().rank() == destinationIndex) {
		for(auto&&x : {mpiWorld().rank(), mpiWorld().rank() + 1}) data.emplace_back(createPODtypeForRank(x));
	}
	const int dataSize = 2;
	receiveCounts[sourceIndex] = dataSize;
	receiveCounts[destinationIndex] = dataSize;
	displacements[sourceIndex] = dataSize;
	displacements[destinationIndex] = 0;

	const std::vector<PODtype> gathered = mpiWorld().varyingAllGather(data,receiveCounts,displacements);
	const std::vector<int> expectedOrder = { destinationIndex, destinationIndex +1, sourceIndex, sourceIndex + 1};
	if(mpiWorld().size() == 1) ASSERT_EQ(data.size(),gathered.size());
	else ASSERT_EQ(expectedOrder.size(),gathered.size());
	for(unsigned i = 0; i < gathered.size(); ++i) {
		expectNear(createPODtypeForRank(expectedOrder[i]), gathered[i], defaultTolerance);
	}
}


TEST_F(NiceMPItests, asyncSendDoNotBlock) {
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) mpiWorld().asyncSend(podTypeInstance,destinationIndex);
	SUCCEED();
}
TEST_F(NiceMPItests, asyncReceiveDoNotBlock) {
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) mpiWorld().asyncReceive<PODtype>(sourceIndex);
	SUCCEED();
}
TEST_F(NiceMPItests, asyncSendAndReceiveAndWait) {
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) {
		SendRequest r = mpiWorld().asyncSend(podTypeInstance,destinationIndex);
		r.wait();
	}
	if(mpiWorld().rank() == destinationIndex) {
		ReceiveRequest<PODtype> r = mpiWorld().asyncReceive<PODtype>(sourceIndex);
		r.wait();
		std::vector<PODtype> data = r.take();
		ASSERT_EQ(1,data.size());
		expectNear(podTypeInstance, data[0], defaultTolerance);
	}
}
TEST_F(NiceMPItests, asyncSendAndReceiveWithTag) {
	const int tag = 3;
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) {
		SendRequest r = mpiWorld().asyncSend(podTypeInstance,destinationIndex,tag);
		r.wait();
	}
	if(mpiWorld().rank() == destinationIndex) {
		ReceiveRequest<PODtype> r = mpiWorld().asyncReceive<PODtype>(sourceIndex,MPI_ANY_TAG);
		r.wait();
		std::vector<PODtype> data = r.take();
		ASSERT_EQ(1,data.size());
		expectNear(podTypeInstance, data[0], defaultTolerance);
	}
}
TEST_F(NiceMPItests, asyncSendAndReceiveAndTest) {
	if(sourceIndex == destinationIndex) return;
	if(mpiWorld().rank() == sourceIndex) {
		SendRequest r = mpiWorld().asyncSend(podTypeInstance,destinationIndex);
		while(!r.isCompleted()) std::this_thread::sleep_for(std::chrono::microseconds{});
	}
	if(mpiWorld().rank() == destinationIndex) {
		ReceiveRequest<PODtype> r = mpiWorld().asyncReceive<PODtype>(sourceIndex);
		while(!r.isCompleted()) std::this_thread::sleep_for(std::chrono::microseconds{});
		std::vector<PODtype> data = r.take();
		ASSERT_EQ(1,data.size());
		expectNear(podTypeInstance, data[0], defaultTolerance);
	}
}


TEST_F(NiceMPItests, sendAndReceiveAnythingVector) {
	testSendAndReceiveCollection<std::vector<PODtype>>();
}
TEST_F(NiceMPItests, sendAndReceiveAnythingArray) {
	testSendAndReceiveCollection<std::array<PODtype,2>>();
}
TEST_F(NiceMPItests, broadcastVector) {
	testBroadcastCollection<std::vector<PODtype>>();
}
TEST_F(NiceMPItests, broadcastArray) {
	testBroadcastCollection<std::array<PODtype,2>>();
}
TEST_F(NiceMPItests, gatherVector) {
	testGather<std::vector<PODtype>>();
}
TEST_F(NiceMPItests, gatherArray) {
	testGather<std::array<PODtype,2>>();
}
TEST_F(NiceMPItests, allGatherVector) {
	testAllGather<std::vector<PODtype>>();
}
TEST_F(NiceMPItests, allGatherArray) {
	testAllGather<std::array<PODtype,2>>();
}
TEST_F(NiceMPItests, asyncSendAndReceiveAndWaitVector) {
	testAsyncSendAndReceiveCollection<std::vector<PODtype>>();
}
TEST_F(NiceMPItests, asyncSendAndReceiveAndWaitArray) {
	testAsyncSendAndReceiveCollection<std::array<PODtype,2>>();
}
