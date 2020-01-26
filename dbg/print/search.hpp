#ifndef DBG_SEARCH_HPP
#define DBG_SEARCH_HPP

#include "query/sindex.hpp"

#include "dbg/print/tree.hpp"

void visualize (std::ostream& out, const query::search::OpTrieT& otrie)
{
	auto root = otrie.root();
	std::unordered_map<const query::search::OpTrieT::NodeT*,std::string>
	labels = {{root, "<ROOT>"}};
	PrettyTree<const query::search::OpTrieT::NodeT*> drawer(
		[&labels](const query::search::OpTrieT::NodeT*& root)
		{
			auto& children = root->children_;
			query::PathNodesT keys;
			keys.reserve(children.size());
			for (auto& child : children)
			{
				keys.push_back(child.first);
			}
			std::sort(keys.begin(), keys.end(),
				[](query::PathNode& l, query::PathNode& r)
				{
					if (l.op_ == r.op_)
					{
						return l.idx_ < r.idx_;
					}
					return egen::name_op(l.op_) < egen::name_op(r.op_);
				});
			std::vector<const query::search::OpTrieT::NodeT*> out;
			out.reserve(keys.size());
			for (auto& key : keys)
			{
				auto& next = children.at(key);
				out.push_back(next);
				labels.emplace(next, egen::name_op(key.op_) +
					":" + fmts::to_string(key.idx_));
			}
			return out;
		},
		[&labels](std::ostream& out, const query::search::OpTrieT::NodeT* root)
		{
			out << estd::must_getf(labels, root, "unlabelled node %p", root);
			if (root->leaf_.has_value())
			{
				const query::search::PathVal& val = *root->leaf_;
				if (val.leaves_.size() > 0)
				{
					std::vector<std::string> leaves;
					leaves.reserve(val.leaves_.size());
					for (const auto& ls : val.leaves_)
					{
						leaves.push_back(ls.first->to_string());
					}
					std::sort(leaves.begin(), leaves.end());
					out << ",leaves:" << fmts::to_string(leaves.begin(), leaves.end());
				}
			}
		});
	drawer.print(out, root);
}

#endif // DBG_SEARCH_HPP
