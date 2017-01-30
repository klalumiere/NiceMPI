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
#include <NiceMPI/private/MPIcommunicatorHandle.h>

using namespace NiceMPI;

class MPIcommunicatorHandleTests : public ::testing::Test {
public:
	bool areCongruentMPI(const MPI_Comm &a, const MPI_Comm &b) const {
		int result;
		MPI_Comm_compare(a, b, &result);
		return result == MPI_CONGRUENT;
	}
	bool areIdenticalMPI(const MPI_Comm &a, const MPI_Comm &b) const {
		int result;
		MPI_Comm_compare(a, b, &result);
		return result == MPI_IDENT;
	}

	const MPIcommunicatorHandle world{MPI_COMM_WORLD};
};


TEST_F(MPIcommunicatorHandleTests, MPIcommunicatorHandleIsCongruent) {
	EXPECT_TRUE( areCongruentMPI(MPI_COMM_WORLD, world.get()) );
}
TEST_F(MPIcommunicatorHandleTests, MPIcommunicatorHandleIsIdentical) {
	MPI_Comm notRvalue = MPI_COMM_SELF;
	MPIcommunicatorHandle x(&notRvalue);
	EXPECT_TRUE( areIdenticalMPI(MPI_COMM_SELF, x.get()) );
}
TEST_F(MPIcommunicatorHandleTests, Copy) {
	const MPIcommunicatorHandle copy{world};
	EXPECT_TRUE( areCongruentMPI(world.get(),copy.get()) );
}
TEST_F(MPIcommunicatorHandleTests, Move) {
	MPIcommunicatorHandle toMove(MPI_COMM_SELF);
	MPI_Comm expected = toMove.get();
	const MPIcommunicatorHandle movedInto{std::move(toMove)};
	EXPECT_TRUE( areIdenticalMPI(expected, movedInto.get()) );
}


TEST_F(MPIcommunicatorHandleTests, Assignment) {
	const MPIcommunicatorHandle self(MPI_COMM_SELF);
	MPIcommunicatorHandle x{world};
	EXPECT_TRUE( areCongruentMPI(self.get(),(x = self).get()) );
}
TEST_F(MPIcommunicatorHandleTests, SelfAssignment) {
	MPIcommunicatorHandle x{world};
	x = x;
	SUCCEED();
}
TEST_F(MPIcommunicatorHandleTests, MoveAssignment) {
	MPIcommunicatorHandle self(MPI_COMM_SELF);
	MPI_Comm lhs = self.get();
	MPIcommunicatorHandle x{world};
	EXPECT_TRUE( areIdenticalMPI(lhs,(x = std::move(self)).get()) );
}
