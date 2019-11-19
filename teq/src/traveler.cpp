#include "teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

struct OwnerTracker final : public OnceTraveler
{
	OwnerMapT owners_;

private:
	/// Implementation of OnceTraveler
	void visit_leaf (iLeaf* leaf) override {}

	/// Implementation of OnceTraveler
	void visit_func (iFunctor* func) override
	{
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			TensptrT tens = child.get_tensor();
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
