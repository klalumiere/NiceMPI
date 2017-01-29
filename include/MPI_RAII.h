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

#ifndef MPI_RAII_H
#define MPI_RAII_H

#include <exception> // std::terminate
#include <mpi.h> // MPI_Init
#include "NiceMPIexception.h" // handleError

namespace NiceMPI {

/** \brief Initialize and finalize MPI using
	[RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization). */
struct MPI_RAII {
	/** \brief Initializes MPI. */
	MPI_RAII(int argc, char* argv[]) {
		handleError(MPI_Init(&argc, &argv));
	}
	/** \brief Finalizes MPI. */
	~MPI_RAII() {
		int error = MPI_Finalize();
		if(error != MPI_SUCCESS) std::terminate();
	}
};

} // NiceMPi

#endif  /* MPI_RAII_H */
