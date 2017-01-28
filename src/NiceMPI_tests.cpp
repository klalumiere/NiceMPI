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

#include <vector>
#include <gtest/gtest.h>
#include <buildInformationNiceMPI.h>
#include <NiceMPI.h>

using namespace NiceMPI;

class NiceMPItests : public ::testing::Test {
public:
	struct PODtype {
		int theInt;
		double theDouble;
		char theChar;
	};

	bool areCongruent(const MPI_Comm &a, const MPI_Comm &b) const {
		int result;
		MPI_Comm_compare(a, b, &result);
		return result == MPI_CONGRUENT;
	}
	bool areEquals(const MPI_Comm &a, const MPI_Comm &b) const {
		int result;
		MPI_Comm_compare(a, b, &result);
		return result == MPI_IDENT;
	}
	Communicator splitEven() const {
		const int color = world.rank() % 2;
		const int key = world.rank();
		return world.split(color,key);
	}
	void expectNear(const PODtype& expected, const PODtype& actual, double tolerance) {
		EXPECT_EQ(expected.theInt, actual.theInt);
		EXPECT_NEAR(expected.theDouble, actual.theDouble, tolerance);
		EXPECT_EQ(expected.theChar, actual.theChar);
	}
	PODtype createPODtypeForRank(int rank) {
		PODtype result{podTypeInstance};
		result.theInt = rank*2;
		return result;
	}
	std::vector<PODtype> createPODtypeCollection(const int count) {
		if(mpiWorld().rank() != sourceIndex) return {};
		std::vector<PODtype> result;
		for(int i = 0; i < count; ++i) result.push_back(createPODtypeForRank(i));
		return result;
	}
	void testGather(const std::vector<PODtype>& gathered) {
		ASSERT_EQ(mpiWorld().size(),gathered.size());
		for(int i=0; i<mpiWorld().size(); ++i) {
			PODtype expected{podTypeInstance};
			expected.theInt = 2*i;
			expectNear(expected,gathered[i],defaultTolerance);
		}
	}

	const Communicator world;
	const int sourceIndex = 0;
	const int destinationIndex = world.size() -1;
	const double defaultTolerance = 1e-10;
	const PODtype podTypeInstance{42,6.66,'K'};
	const std::vector<PODtype> defaultCollection = createPODtypeCollection(mpiWorld().size());
};


TEST_F(NiceMPItests, NiceMPIexceptionExist) {
	const NiceMPIexception x{3};
	EXPECT_EQ(3,x.error);
}
TEST_F(NiceMPItests, newCommunicatorIsWorldCongruent) {
	EXPECT_TRUE( areCongruent(MPI_COMM_WORLD,world.get()) );
	EXPECT_FALSE( areEquals(MPI_COMM_WORLD,world.get()) );
}
TEST_F(NiceMPItests, Copy) {
	const Communicator copy{world};
	EXPECT_TRUE( areCongruent(world.get(),copy.get()) );
}
TEST_F(NiceMPItests, Move) {
	Communicator toMove;
	MPI_Comm lhs = toMove.get();
	const Communicator movedInto{std::move(toMove)};
	EXPECT_TRUE( areEquals(lhs,movedInto.get()) );
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
	const Communicator splitted = splitEven();

	int expectedSize = 0;
	if(maxWolrdSize % 2 == 0) expectedSize = std::max(world.size()/2,1);
	else if(world.rank() % 2 == 0) expectedSize = world.size()/2 + 1;
	else expectedSize = world.size()/2;

	EXPECT_EQ(expectedSize,splitted.size());
	EXPECT_EQ(world.rank()/2,splitted.rank());
}
TEST_F(NiceMPItests, Assignment) {
	const Communicator splitted = splitEven();
	Communicator worldCopy{world};
	EXPECT_TRUE( areCongruent(splitted.get(),(worldCopy = splitted).get()) );
}
TEST_F(NiceMPItests, SelfAssignment) {
	Communicator worldCopy;
	worldCopy = worldCopy;
	SUCCEED();
}
TEST_F(NiceMPItests, MoveAssignment) {
	Communicator splitted = splitEven();
	MPI_Comm lhs = splitted.get();
	Communicator worldCopy;
	EXPECT_TRUE( areEquals(lhs,(worldCopy = std::move(splitted)).get()) );
}
TEST_F(NiceMPItests, mpiWorld) {
	EXPECT_TRUE( areEquals(MPI_COMM_WORLD,mpiWorld().get()) );
	EXPECT_TRUE( areEquals(mpiWorld().get(),mpiWorld().get()) );
	EXPECT_TRUE( areCongruent(world.get(),mpiWorld().get()) );
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
	if(mpiWorld().rank()==sourceIndex) testGather(gathered);
	else EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, allGather) {
	const PODtype myData = createPODtypeForRank(mpiWorld().rank());
	testGather(mpiWorld().allGather(myData));
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
TEST_F(NiceMPItests, varyingScatterHalfToTheSame) {
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
	if(mpiWorld().rank()==sourceIndex) testGather(gathered);
	else EXPECT_EQ(0,gathered.size());
}
TEST_F(NiceMPItests, varyingGatherHalfFromTheSame) {
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
		for(unsigned i = 0; i < gathered.size(); ++i) {
			expectNear(createPODtypeForRank(expectedOrder[i]), gathered[i], defaultTolerance);
		}
	}
	else EXPECT_EQ(0,gathered.size());
}
