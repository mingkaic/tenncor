#include "tag/tag.hpp"

#ifndef TAG_GROUPY_HPP
#define TAG_GROUPY_HPP

namespace tag
{

struct Grouper final : public ade::iTraveler
{
	Grouper (std::string group, std::unordered_set<ade::iTensor*> stops) :
		group_(group), stops_(stops.begin(), stops.end()) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (stops_.end() == stops_.find(leaf))
		{
			tag::group_tag(leaf, group_);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (stops_.end() == stops_.find(func))
		{
			auto& children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}
			tag::group_tag(func, group_);
		}
	}

	std::string group_;

	std::unordered_set<ade::iTensor*> stops_;
};

}

#endif // TAG_GROUPY_HPP
