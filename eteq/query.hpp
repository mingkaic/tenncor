#ifndef ETEQ_QUERY_HPP
#define ETEQ_QUERY_HPP

#include "teq/traveler.hpp"

namespace eteq
{

struct LayerQuery final : public teq::iOnceTraveler
{
	teq::TensT found_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		auto children = func.get_children();
		for (auto child : children)
		{
			child->accept(*this);
		}

		if (auto lattr = func.get_attr(teq::layer_key))
		{
			found_.push_back(&func);
		}
	}
};

}

#endif // ETEQ_QUERY_HPP
