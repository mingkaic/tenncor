#ifndef DBG_SEARCH_HPP
#define DBG_SEARCH_HPP

#include "query/sindex.hpp"

#include "dbg/print/tree.hpp"

void visualize (std::ostream& out, const query::search::OpTrieT& otrie,
	size_t ndepth = std::numeric_limits<size_t>::max())
{
	auto root = otrie.root();
	std::unordered_map<const query::search::OpTrieT::NodeT*,std::string>
	labels = {{root, "<ROOT>"}};
	PrettyTree<const query::search::OpTrieT::NodeT*> drawer(
		[&labels, ndepth](const query::search::OpTrieT::NodeT*& root, size_t depth)
		{
			std::vector<const query::search::OpTrieT::NodeT*> out;
			if (depth >= ndepth)
			{
				return out;
			}
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
					if (l.opname_ == r.opname_)
					{
						return l.idx_ < r.idx_;
					}
					return l.opname_ < r.opname_;
				});
			out.reserve(keys.size());
			for (auto& key : keys)
			{
				auto& next = children.at(key);
				out.push_back(next);
				labels.emplace(next, key.opname_ +
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
					out << ",leaves:[" << fmts::join(",", leaves.begin(), leaves.end()) << "]";
				}
				if (val.attrs_.size() > 0)
				{
					std::vector<std::string> attrs;
					attrs.reserve(val.attrs_.size());
					for (auto& apair : val.attrs_)
					{
						auto attrkeys = apair.first->ls_attrs();
						std::sort(attrkeys.begin(), attrkeys.end());
						attrs.push_back(
							fmts::to_string(attrkeys.begin(), attrkeys.end()));
					}
					std::sort(attrs.begin(), attrs.end());
					out << ",attrs:[" << fmts::join(",", attrs.begin(), attrs.end()) << "]";
				}
				if (val.comms_.size() > 0)
				{
					std::vector<std::string> comms;
					comms.reserve(val.comms_.size());
					for (auto& cpair : val.comms_)
					{
						comms.push_back(cpair.first->to_string());
					}
					std::sort(comms.begin(), comms.end());
					out << ",comms:[" << fmts::join(",", comms.begin(), comms.end()) << "]";
				}
			}
		});
	drawer.print(out, root);
}

#endif // DBG_SEARCH_HPP
