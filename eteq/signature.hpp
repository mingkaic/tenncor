#include "teq/ishape.hpp"
#include "teq/itensor.hpp"

#include "eigen/generated/dtype.hpp"

#ifndef ETEQ_SIGNATURE_HPP
#define ETEQ_SIGNATURE_HPP

namespace eteq
{

struct iSignature
{
	virtual ~iSignature (void) = default;

	virtual teq::TensptrT build_tensor (void) const = 0;

	virtual teq::ShapeSignature shape_sign (void) const = 0;
};

}

#endif // ETEQ_SIGNATURE_HPP
