#include "ade/ade.hpp"

namespace experimental
{

using DistanceMapT = std::unordered_map<ade::iTensor*,size_t>;

using EdgeDistanceMapT = std::unordered_map<ade::iTensor*,ade::DistanceMapT>;

struct DistanceFinder final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (distances_.end() == distances_.find(leaf))
		{
			distances_.emplace(leaf, DistanceMapT{{leaf, 0}});
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (distances_.end() == distances_.find(leaf))
		{
			DistanceMapT distmap = {{func, 0}};
			auto& children = func->get_children();
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
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
