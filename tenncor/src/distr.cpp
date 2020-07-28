
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

void set_distmgr (distr::iDistMgrptrT mgr)
{
	set_distmgr(mgr, eigen::global_context());
}

}

#endif
