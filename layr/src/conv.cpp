#include "layr/conv.hpp"

#ifdef LAYR_CONV_HPP

namespace layr
{

LayerptrT ConvBuilder::build (void) const
{
	if (nullptr == weight_)
	{
		logs::fatal("cannot build conv with null weight");
	}
	return std::make_shared<Conv>(weight_, bias_, label_);
}

}

#endif
