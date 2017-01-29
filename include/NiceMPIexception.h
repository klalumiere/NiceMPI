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

#ifndef NICEMPIEXCEPTION_H
#define NICEMPIEXCEPTION_H

#include <stdexcept> // std::runtime_error
#include <string> // std::to_string
#include <mpi.h> // MPI_SUCCESS

namespace NiceMPI {

/** \brief Every exceptions inherit from this class. */
class NiceMPIexception: public std::runtime_error {
public:
	/** \brief Create the exception with the error code \p error. */
	NiceMPIexception(int error): std::runtime_error("Error code " + std::to_string(error) + " in MPI."), error(error)
	{}

	/** \brief Exception error code. */
	const int error;
};

/** \brief Turns MPI \p error in \p NiceMPIexception. */
inline void handleError(int error) {
	if(error != MPI_SUCCESS) throw NiceMPIexception{error};
}

} // NiceMPi

#endif  /* NICEMPIEXCEPTION_H */
