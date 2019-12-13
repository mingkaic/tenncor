#include "teq/ishape.hpp"
#include "teq/idata.hpp"

#ifndef TEQ_SIGNATURE_HPP
#define TEQ_SIGNATURE_HPP

namespace teq
{

struct iSignature
{
	virtual ~iSignature (void) = default;

	virtual bool can_build (void) const = 0;

	/// Return data node if build is successful, otherwise return nullptr
	virtual DataptrT build_data (void) const = 0;

	virtual ShapeSignature shape_sign (void) const = 0;
};

}

#endif // TEQ_SIGNATURE_HPP
