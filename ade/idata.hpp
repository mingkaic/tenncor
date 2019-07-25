///
/// ileaf.hpp
/// ade
///
/// Purpose:
/// Define common interface for node unveiling data information
///

#include <cstdlib>
#include <string>

#ifndef ADE_IDATA_HPP
#define ADE_IDATA_HPP

namespace ade
{

/// Interface for unveiling data
struct iData
{
	virtual ~iData (void) = default;

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;

	/// Return data type label (for better readability)
	virtual std::string type_label (void) const = 0;

	/// Return number of bytes in the data
	virtual size_t nbytes (void) const = 0;
};

}

#endif // ADE_IDATA_HPP
