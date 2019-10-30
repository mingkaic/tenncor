#include "layr/dense.hpp"

#ifdef LAYR_DENSE_HPP

namespace layr
{

LayerptrT DenseBuilder::build (void) const
{
	if (nullptr == weight_)
	{
		logs::fatal("cannot build dense with null weight");
	}
	return std::make_shared<Dense>(weight_, bias_, params_, label_);
}

}

#endif
