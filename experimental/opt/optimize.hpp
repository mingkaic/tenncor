#ifndef EXPERIMENTAL_OPT_OPTIMIZE_HPP
#define EXPERIMENTAL_OPT_OPTIMIZE_HPP

#include "dbg/print/search.hpp"

#include "query/sindex.hpp"

namespace opt
{

template <typename T>
void set_intersection (std::unordered_set<T>& out,
	const std::unordered_set<T>& a,
	const std::unordered_set<T>& b)
{
	out.clear();
	for (const auto & e : a)
	{
		if (estd::has(b, e))
		{
			out.emplace(e);
		}
	}
}

void filter_path (query::search::OpTrieT::NodeT* sroot,
	const teq::TensSetT& rset)
{
	auto& nexts = sroot->children_;
	for (auto it = nexts.begin(), et = nexts.end(); it != et;)
	{
		auto& next = it->second;
		filter_path(next, rset);
		if (next->children_.empty() && false == next->leaf_.has_value())
		{
			delete next;
			it = nexts.erase(it);
		}
		else
		{
			++it;
		}
	}

	if (sroot->leaf_.has_value())
	{
		teq::TensSetT isec;
		auto& leaves = sroot->leaf_->leaves_;
		for (auto it = leaves.begin(), et = leaves.end(); it != et;)
		{
			set_intersection(isec, rset, it->second);
			if (isec.size() > 0)
			{
				it->second = isec;
				++it;
			}
			else
			{
				// delete
				it = leaves.erase(it);
			}
		}
		auto& attrs = sroot->leaf_->attrs_;
		for (auto it = attrs.begin(), et = attrs.end(); it != et;)
		{
			set_intersection(isec, rset, it->second);
			if (isec.size() > 0)
			{
				it->second = isec;
				++it;
			}
			else
			{
				// delete
				it = attrs.erase(it);
			}
		}
		if (leaves.empty() && attrs.empty())
		{
			sroot->leaf_.reset();
		}
	}
}

using OpnodeT = query::search::OpTrieT::NodeT;

using OpnodesT = std::unordered_set<const OpnodeT*>;

void reverse_triemap (std::unordered_map<OpnodeT*,OpnodeT*>& triemap,
	const OpnodeT* root)
{
	for (auto child : root->children_)
	{
		auto& c = child->second;
		triemap.emplace(c, root);
		reverse_triemap(triemap, c);
	}
}

template <typename T>
void leaf_triemap(
	std::unordered_map<eteq::Constant<T>*,OpnodesT>& cstmap,
	std::unordered_map<eteq::iLeaf<T>*,OpnodesT>& lefmap,
	const OpnodeT* root)
{
	for (auto child : root->children_)
	{
		leaf_triemap(cstmap, lefmap, child->second);
	}
	if (root->leaf_.has_value())
	{
		for (const auto& leaf : root->leaf_.leaves_)
		{
			if (auto cst = dynamic_cast<eteq::Constant<T>*>(leaf.first))
			{
				cstmap[cst].emplace(root);
			}
			else
			{
				lefmap[static_cast<eteq::iLeaf<T>*>(leaf.first)].emplace(root);
			}
		}
	}
}

template <typename T>
void optimize (const eteq::ETensorsT<T>& roots, )
{
	if (roots.empty())
	{
		return;
	}

	// model entire graph using trie
	query::search::OpTrieT sindex;
	query::search::populate_itable(sindex,
		teq::TensptrsT(roots.begin(), roots.end()));



	// OpnodeT* sroot = sindex.root();
	// std::unordered_map<OpnodeT*,OpnodeT*> reverse_trie;
	// reverse_triemap(reverse_trie, sroot);

	// std::unordered_map<eteq::Constant<T>*,OpnodesT> cstmap;
	// std::unordered_map<eteq::iLeaf<T>*,OpnodesT> lefmap;
	// leaf_triemap(cstmap, lefmap, sroot);

	// // only select leaf/attr paths going to roots
	// teq::TensSetT rset;
	// std::transform(roots.begin(), roots.end(),
	// 	std::inserter(rset, rset.end()),
	// 	[](teq::TensptrT tens) { return tens.get(); });
	// filter_path(sroot, rset);

	std::stringstream ss;
	visualize(ss, sindex, 3);
for (auto root : roots)
{
std::cout << root->to_string() << std::endl;
}
std::cout << ss.str() << std::endl;
}

}

#endif // EXPERIMENTAL_OPT_OPTIMIZE_HPP
