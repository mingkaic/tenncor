#include "rocnnet/modl/rbm.hpp"

#ifdef MODL_RBM_HPP

namespace modl
{

LayerptrT RBMBuilder::build (void) const
{
	if (nullptr == weight_)
	{
		logs::fatal("cannot build rbm with null weight");
	}
	return std::make_shared<RBM>(weight_, hbias_, vbias_, label_);
}

}

#endif
