///
/// ileaf.hpp
/// teq
///
/// Purpose:
/// Define common interface for node unveiling data information
///

#include <cstdlib>
#include <string>

#include "teq/shape.hpp"

#ifndef TEQ_IDATA_HPP
#define TEQ_IDATA_HPP

namespace teq
{

/// Interface for unveiling data
struct iData
{
	virtual ~iData (void) = default;

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;

	/// Return the shape of the data
	virtual Shape data_shape (void) const = 0;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;

	/// Return data type label (for better readability)
	virtual std::string type_label (void) const = 0;

	/// Return number of bytes in the data
	virtual size_t nbytes (void) const = 0;
};

using DataptrT = std::shared_ptr<iData>;

using DatasT = std::vector<DataptrT>;

}

#endif // TEQ_IDATA_HPP
