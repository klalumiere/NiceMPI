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
