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

#ifndef MPICOMMUNICATORHANDLE_HPP
#define MPICOMMUNICATORHANDLE_HPP

#include <exception> // std::terminate
#include <utility> // std::move
#include "NiceMPIexception.h" // handleError

namespace NiceMPI {

/** \brief Interface of an implementation of a MPIcommunicatorHandle. */
class MPIcommunicatorHandleImpl {
public:
	/** \brief This class is an interface, hence, virtual destructor is requierd. */
	virtual ~MPIcommunicatorHandleImpl()
	{}
	/** \brief Returns a deep copy of this object. */
	virtual std::unique_ptr<MPIcommunicatorHandleImpl> deepCopy() const = 0;
	/** \brief Returns the underlying MPI_Comm implementation. */
	virtual MPI_Comm get() const = 0;
	/** \brief Returns the underlying MPI_Comm implementation. */
	virtual MPI_Comm get() = 0;
};

/** \brief Implements a MPI communicator handle that owns the MPI_Comm. */
class OwnedCommunicator: public MPIcommunicatorHandleImpl {
public:
	/** \brief Creates a communicator handle congruent (but not identical) to rhs. */
	OwnedCommunicator(MPI_Comm rhs) {
		handleError(MPI_Comm_dup(rhs,&mpiCommunicator));
	}
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator(const OwnedCommunicator&) = delete;
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator(OwnedCommunicator&&) = delete;
	/** \brief Destroys the underlying MPI communicator. */
	~OwnedCommunicator() {
		int error = MPI_Comm_free(&mpiCommunicator);
		if(error != MPI_SUCCESS) std::terminate();
	}
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator& operator=(const OwnedCommunicator&) = delete;
	/** \brief [Rule of 5](http://en.cppreference.com/w/cpp/language/rule_of_three). */
	OwnedCommunicator& operator=(OwnedCommunicator&&) = delete;
	std::unique_ptr<MPIcommunicatorHandleImpl> deepCopy() const override {
		return std::unique_ptr<MPIcommunicatorHandleImpl>(new OwnedCommunicator(mpiCommunicator));
	}
	MPI_Comm get() const override {
		return mpiCommunicator;
	}
	MPI_Comm get() override {
		return mpiCommunicator;
	}

private:
	/** \brief Underlying MPI implementation of this handle. */
	MPI_Comm mpiCommunicator;	
};

/** \brief Implements a MPI communicator handle that is only a proxy for the MPI_Comm. */
class ProxyCommunicator: public MPIcommunicatorHandleImpl {
public:
	/** \brief Creates a communicator handle identical to rhs. */
	ProxyCommunicator(MPI_Comm rhs): mpiCommunicator(rhs)
	{}
	std::unique_ptr<MPIcommunicatorHandleImpl> deepCopy() const override {
		return std::unique_ptr<MPIcommunicatorHandleImpl>(new OwnedCommunicator(mpiCommunicator)); // Can't copy proxy
	}
	MPI_Comm get() const override {
		return mpiCommunicator;
	}
	MPI_Comm get() override {
		return mpiCommunicator;
	}

private:
	/** \brief Underlying MPI implementation of this handle. */
	MPI_Comm mpiCommunicator;
};

inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPI_Comm mpiCommunicator)
: impl(new OwnedCommunicator(mpiCommunicator))
{}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPI_Comm* mpiCommunicator)
: impl(new ProxyCommunicator(*mpiCommunicator))
{}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(const MPIcommunicatorHandle& rhs) {
	impl = rhs.impl->deepCopy();
}
inline MPIcommunicatorHandle::MPIcommunicatorHandle(MPIcommunicatorHandle&& rhs) {
	impl = std::move(rhs.impl);
}
inline MPIcommunicatorHandle::~MPIcommunicatorHandle() = default;
inline MPIcommunicatorHandle& MPIcommunicatorHandle::operator=(const MPIcommunicatorHandle& rhs) {
	impl = rhs.impl->deepCopy();
	return *this;
}
inline MPIcommunicatorHandle& MPIcommunicatorHandle::operator=(MPIcommunicatorHandle&& rhs) {
	impl = std::move(rhs.impl);
	return *this;
}
inline MPI_Comm MPIcommunicatorHandle::get() const {
	return impl->get();
}
inline MPI_Comm MPIcommunicatorHandle::get() {
	return impl->get();
}

} // NiceMPi

#endif  /* MPICOMMUNICATORHANDLE_HPP */
