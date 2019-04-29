#include "ade/traveler.hpp"

#ifdef ADE_TRAVELER_HPP

namespace ade
{

struct OwnerTracker final : public iTraveler
{
	/// Implementation of iTraveler
	void visit (iLeaf* leaf) override {}

	/// Implementation of iTraveler
	void visit (iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func))
		{
			auto& children = func->get_children();
			for (auto& child : children)
			{
				TensptrT tens = child.get_tensor();
				tens->accept(*this);
				owners_.emplace(tens.get(), tens);
			}
			visited_.emplace(func);
		}
	}

	/// Map of parent nodes in path
	std::unordered_set<iFunctor*> visited_;

	OwnerMapT owners_;
};

OwnerMapT track_owners (TensptrT root)
{
	OwnerTracker tracker;
	root->accept(tracker);
	tracker.owners_.emplace(root.get(), root);
	return tracker.owners_;
}

}

#endif
