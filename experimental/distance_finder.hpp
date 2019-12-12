#include "teq/teq.hpp"

namespace experimental
{

using DistanceMapT = std::unordered_map<teq::iTensor*,size_t>;

using EdgeDistanceMapT = std::unordered_map<teq::iTensor*,teq::DistanceMapT>;

struct DistanceFinder final : public teq::iTraveler
{
	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(distances_, leaf))
		{
			distances_.emplace(leaf, DistanceMapT{{leaf, 0}});
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(distances_, func))
		{
			DistanceMapT distmap = {{func, 0}};
			auto children = func->get_children();
			for (teq::TensptrT tens : children)
			{
				tens->accept(*this);
				DistanceMapT& subdistance = distances_[tens.get()];
				for (auto distpair : subdistance)
				{
					size_t mindist = distpair.second;
					auto it = distmap.find(distpair);
					if (distmap.end() != it && it.second < distpair.second)
					{
						mindist = it.second;
					}
					distmap[distpair.first] = mindist;
				}
			}
			distances_.emplace(func, distmap);
		}
	}

	EdgeDistanceMapT distances_;
};

}
