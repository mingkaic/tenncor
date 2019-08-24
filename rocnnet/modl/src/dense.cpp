#include "rocnnet/modl/dense.hpp"

#ifdef MODL_DENSE_HPP

namespace modl
{

LayerptrT DenseBuilder::build (void) const
{
	return std::make_shared<Dense>(weight_, bias_, label_);
}

}

#endif
