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

#include <mpi.h>

/** \brief An alternative to Boost.MPI for a user friendly C++ interface for MPI (MPICH). **/
namespace NiceMPI {

/** \brief Initialize and finalize MPI using RAII. */
struct MPI_RAII {
	/** \brief Initializes MPI. */
	MPI_RAII(int argc, char* argv[]) {
		MPI_Init(&argc, &argv);
	}
	/** \brief Finalizes MPI. */
	~MPI_RAII() {
		MPI_Finalize();
	}
};

/** \brief Represents a MPI communitator. */
class Communicator {
public:
	Communicator() {
		MPI_Comm_dup(MPI_COMM_WORLD, &mpiCommunicator);
	}
	Communicator(const Communicator& rhs) {
		MPI_Comm_dup(rhs.mpiCommunicator, &mpiCommunicator);
	}
	Communicator(Communicator&& rhs): Communicator() {
		std::swap(mpiCommunicator,rhs.mpiCommunicator);
	}
	~Communicator() {
		MPI_Comm_free(&mpiCommunicator);
	}
	Communicator& operator=(const Communicator& rhs) {
		if(this != &rhs) MPI_Comm_dup(rhs.mpiCommunicator, &mpiCommunicator);
		return *this;
	}
	Communicator& operator=(Communicator&& rhs) {
		std::swap(mpiCommunicator,rhs.mpiCommunicator);
		return *this;
	}

	/** \brief Returns the MPI communicator associated to \p this. This method breaks encapsulation, but it is
  provided in order to facilitate the interface with MPI functions not implemented here. Try to minimize its use. */
	MPI_Comm get() const {
		return mpiCommunicator;
	}
	int rank() const {
		int rank;
		MPI_Comm_rank(mpiCommunicator, &rank);
		return rank;
	}
	int size() const {
		int size;
		MPI_Comm_size(mpiCommunicator, &size);
		return size;
	}
	Communicator split(int color, int key) const {
		MPI_Comm splitted;
		MPI_Comm_split(mpiCommunicator,color,key,&splitted);
		return Communicator{std::move(splitted)};
	}

private:
	Communicator(MPI_Comm&& mpiCommunicatorRhs) {
		mpiCommunicator = mpiCommunicatorRhs;
	}

	MPI_Comm mpiCommunicator;
};

} // NiceMPi

#endif  /* NICEMPI_H */
