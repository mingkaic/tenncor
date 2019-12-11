#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

TagRegistry& get_reg (void)
{
	static TagRegistry registry;
	return registry;
}

using RefMapT = std::unordered_map<teq::iTensor*,teq::TensrefT>;

struct Tagger final : public teq::iTraveler
{
	Tagger (teq::TensSetT stops,
		std::function<void(teq::TensrefT)> tag_op) :
		stops_(stops),
		tag_op_(tag_op) {}

	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(stops_, leaf))
		{
			tag_op_(estd::must_getf(owners_, leaf,
				"failed to get reference to leaf %s in group traveler",
				leaf->to_string().c_str()));
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(stops_, func))
		{
			auto children = func->get_children();
			for (const teq::iEdge& child : children)
			{
				teq::TensptrT tens = child.get_tensor();
				owners_.emplace(tens.get(), tens);
				tens->accept(*this);
			}
			tag_op_(estd::must_getf(owners_, func,
				"failed to get reference to functor %s in group traveler",
				func->to_string().c_str()));
		}
	}

	RefMapT owners_;

	teq::TensSetT stops_;

	std::function<void(teq::TensrefT)> tag_op_;
};

void recursive_tag (teq::TensptrT root,
	teq::TensSetT stops,
	std::function<void(teq::TensrefT)> tag_op)
{
	Tagger tagger(stops, tag_op);
	tagger.owners_.emplace(root.get(), root);
	root->accept(tagger);
}

}

#endif
