#include "teq/ishape.hpp"
#include "teq/itensor.hpp"

#include "eigen/generated/dtype.hpp"

#ifndef ETEQ_SIGNATURE_HPP
#define ETEQ_SIGNATURE_HPP

namespace eteq
{

template <typename T>
struct iSignature
{
	virtual ~iSignature (void) = default;

	size_t type_code (void) const
	{
		return egen::get_type<T>();
	}

	std::string type_label (void) const
	{
		return egen::name_type(egen::get_type<T>());
	}

	virtual teq::ShapeSignature shape_sign (void) const = 0;

	virtual std::string to_string (void) const = 0;

	/// Return true if this instance can produce data
	virtual bool is_real (void) const = 0;
};

}

#endif // ETEQ_SIGNATURE_HPP
