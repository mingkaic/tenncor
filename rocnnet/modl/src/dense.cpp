#include "rocnnet/modl/dense.hpp"

#ifdef MODL_DENSE_HPP

namespace modl
{

LayerptrT DenseBuilder::build (void) const
{
	if (nullptr == weight_)
	{
		logs::fatal("cannot build dense with null weight");
	}
	return std::make_shared<Dense>(weight_, bias_, label_);
}

}

#endif
