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

#ifndef MPICOMMUNICATORHANDLE_H
#define MPICOMMUNICATORHANDLE_H

#include <memory> // std::unique_ptr
#include <mpi.h> // MPI_Comm

namespace NiceMPI {

class MPIcommunicatorHandleImpl;

/** \brief Handles the construction/destruction of a MPI communicator. Implements the
	[rule of zero](http://en.cppreference.com/w/cpp/language/rule_of_three) for the Communicator class. */
class MPIcommunicatorHandle {
public:
	/** \brief Creates a handle that contains a communicator congruent (but not identical) to mpiCommunicator. */
	explicit MPIcommunicatorHandle(MPI_Comm mpiCommunicator);
	/** \brief Creates a handle that contains a proxy communicator identical to \p mpiCommunicator.*/
	explicit MPIcommunicatorHandle(MPI_Comm* mpiCommunicator);
	/** \brief Copies the handle \p rhs. */
	MPIcommunicatorHandle(const MPIcommunicatorHandle& rhs);
	/** \brief Moves the handle \p rhs. */
	MPIcommunicatorHandle(MPIcommunicatorHandle&& rhs);
	/** \brief Destroys the handle \p rhs. Only there because of the
		[rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	~MPIcommunicatorHandle();
	/** \brief Assigns \p this handle the handle \p rhs. */
	MPIcommunicatorHandle& operator=(const MPIcommunicatorHandle& rhs);
	/** \brief Moves the handle \p rhs and assigns it to \p this handle.  */
	MPIcommunicatorHandle& operator=(MPIcommunicatorHandle&& rhs);
	/** \brief Returns the underlying MPI_Comm implementation. */
	MPI_Comm get() const;
	/** \brief Returns the underlying MPI_Comm implementation. */
	MPI_Comm get();

private:
	/** \brief Implementation of this handle. */
	std::unique_ptr<MPIcommunicatorHandleImpl> impl;
};

} // NiceMPi

#endif  /* MPICOMMUNICATORHANDLE_H */
