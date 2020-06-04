#include "teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

struct OwnerTracker final : public iOnceTraveler
{
	OwnerMapT owners_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override {}

	/// Implementation of iOnceTraveler
	void visit_func (iFunctor& func) override
	{
		auto children = func.get_dependencies();
		for (const TensptrT& tens : children)
		{
			tens->accept(*this);
			owners_.emplace(tens.get(), tens);
		}
	}
};

OwnerMapT track_owners (TensptrsT roots)
{
	OwnerTracker tracker;
	for (auto root : roots)
	{
		if (nullptr != root)
		{
			root->accept(tracker);
			tracker.owners_.emplace(root.get(), root);
		}
	}
	return tracker.owners_;
}

}

#endif
