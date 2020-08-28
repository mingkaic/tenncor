
#include "tenncor/layr.hpp"

#ifdef TENNCOR_LAYR_HPP

namespace tcr
{

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input)
{
	return layr::connect(root, input);
}

}

#endif
