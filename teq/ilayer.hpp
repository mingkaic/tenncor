#include "teq/ifunctor.hpp"

#ifndef TEQ_ILAYER_HPP
#define TEQ_ILAYER_HPP

namespace teq
{

struct iLayer : public iFunctor
{
	virtual ~iLayer (void) = default;

	/// Implementation of iTensor
	Shape shape (void) const override
	{
		return get_root()->shape();
	}

	virtual TensptrT get_root (void) const = 0;

	/// Return all leaves representing the layer
	virtual TensptrsT get_storage (void) const = 0;
};

using LayerptrT = std::shared_ptr<iLayer>;

}

#endif // TEQ_ILAYER_HPP
