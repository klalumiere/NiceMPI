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

#include <gtest/gtest.h>
#include <buildInformationNiceMPI.h>
#include <NiceMPI.h>

using namespace NiceMPI;

class NiceMPItests : public ::testing::Test {
public:
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

	const Communicator world;
	const int sourceIndex = 0;
	const int destinationIndex = world.size() -1;
	const double tolerance = 1e-10;
};


TEST_F(NiceMPItests, CommunicatorWorldDefault) {
	EXPECT_TRUE( areCongruent(MPI_COMM_WORLD,world.get()) );
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

TEST_F(NiceMPItests, getGlobalWorld) {
	EXPECT_TRUE( areEquals(getWorld().get(),getWorld().get()) );
	EXPECT_TRUE( areCongruent(world.get(),getWorld().get()) );
}
TEST_F(NiceMPItests, sendAndReceive) {
	if(sourceIndex == destinationIndex) return;
	const unsigned char toSend = 'K';
	if(getWorld().rank() == sourceIndex) getWorld().sendDataTo(toSend,destinationIndex);
	if(getWorld().rank() == destinationIndex) {
		const auto result = getWorld().receiveDataFrom<unsigned char>(sourceIndex);
		
		EXPECT_EQ(toSend,result);
	}
}
TEST_F(NiceMPItests, sendAndReceiveAnything) {
	if(sourceIndex == destinationIndex) return;
	struct MyIntWrapper {
		int value;
		bool initialized;
	};
	struct MyCustomStruct {
		int a;
		double b;
		char c;
		MyIntWrapper d;
	};
	const MyCustomStruct toSend{42,6.66,'K',MyIntWrapper{27,true}};
	if(getWorld().rank() == sourceIndex) getWorld().sendDataTo(toSend,destinationIndex);
	if(getWorld().rank() == destinationIndex) {
		auto result = getWorld().receiveDataFrom<MyCustomStruct>(sourceIndex);
		
		EXPECT_EQ(toSend.a,result.a);
		EXPECT_NEAR(toSend.b,result.b,tolerance);
		EXPECT_EQ(toSend.c,result.c);
		EXPECT_EQ(toSend.d.value,result.d.value);
		EXPECT_EQ(toSend.d.initialized,result.d.initialized);
	}
}
TEST_F(NiceMPItests, sendAndReceiveWithTag) {
	if(sourceIndex == destinationIndex) return;
	const unsigned char toSend = 'K';
	const int tag = 3;
	if(getWorld().rank() == sourceIndex) getWorld().sendDataTo(toSend,destinationIndex,tag);
	if(getWorld().rank() == destinationIndex) {
		const auto result = getWorld().receiveDataFrom<unsigned char>(sourceIndex,MPI_ANY_TAG);
		
		EXPECT_EQ(toSend,result);
	}
}
