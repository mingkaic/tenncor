#include "teq/ifunctor.hpp"
#include "teq/placeholder.hpp"

#ifndef TEQ_ILAYER_HPP
#define TEQ_ILAYER_HPP

namespace teq
{

struct iLayer : public iFunctor, public iSignature
{
	virtual ~iLayer (void) = default;

    virtual TensptrT get_input (void) const = 0;

    virtual TensptrT get_output (void) const = 0;

	/// Return all leaves representing the layer
	virtual TensptrsT get_storage (void) const = 0;
};

using LayerptrT = std::shared_ptr<iLayer>;

}

#endif // TEQ_ILAYER_HPP
