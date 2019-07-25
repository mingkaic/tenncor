#include "opt/ivoter.hpp"

#ifdef OPT_IVOTER_HPP

namespace opt
{

void sort_vargs (VoterArgsT& args)
{
	// sort args
	std::sort(args.begin(), args.end(),
		[](const VoterArg& a, const VoterArg& b)
		{
			if (a.label_ == b.label_)
			{
				if (is_equal(a.shaper_, b.shaper_))
				{
					return lt(a.coorder_, b.coorder_);
				}
				return lt(a.shaper_, b.shaper_);
			}
			return a.label_ < b.label_;
		});
}

}

#endif
