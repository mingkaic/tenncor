#include "experimental/opt/optimize.hpp"

#include "dbg/print/teq.hpp"

#ifdef EXPERIMENTAL_OPT_OPTIMIZE_HPP

namespace opt
{

// template <typename T>
// void set_intersection (std::unordered_set<T>& out,
// 	const std::unordered_set<T>& a,
// 	const std::unordered_set<T>& b)
// {
// 	out.clear();
// 	for (const auto & e : a)
// 	{
// 		if (estd::has(b, e))
// 		{
// 			out.emplace(e);
// 		}
// 	}
// }

// void filter_path (query::search::OpTrieT::NodeT* sroot,
// 	const teq::TensSetT& rset)
// {
// 	auto& nexts = sroot->children_;
// 	for (auto it = nexts.begin(), et = nexts.end(); it != et;)
// 	{
// 		auto& next = it->second;
// 		filter_path(next, rset);
// 		if (next->children_.empty() && false == next->leaf_.has_value())
// 		{
// 			delete next;
// 			it = nexts.erase(it);
// 		}
// 		else
// 		{
// 			++it;
// 		}
// 	}

// 	if (sroot->leaf_.has_value())
// 	{
// 		teq::TensSetT isec;
// 		auto& leaves = sroot->leaf_->leaves_;
// 		for (auto it = leaves.begin(), et = leaves.end(); it != et;)
// 		{
// 			set_intersection(isec, rset, it->second);
// 			if (isec.size() > 0)
// 			{
// 				it->second = isec;
// 				++it;
// 			}
// 			else
// 			{
// 				// delete
// 				it = leaves.erase(it);
// 			}
// 		}
// 		auto& attrs = sroot->leaf_->attrs_;
// 		for (auto it = attrs.begin(), et = attrs.end(); it != et;)
// 		{
// 			set_intersection(isec, rset, it->second);
// 			if (isec.size() > 0)
// 			{
// 				it->second = isec;
// 				++it;
// 			}
// 			else
// 			{
// 				// delete
// 				it = attrs.erase(it);
// 			}
// 		}
// 		if (leaves.empty() && attrs.empty())
// 		{
// 			sroot->leaf_.reset();
// 		}
// 	}
// }

// using OpnodeT = query::search::OpTrieT::NodeT;

// using OpnodesT = std::unordered_set<const OpnodeT*>;

// void reverse_triemap (std::unordered_map<OpnodeT*,OpnodeT*>& triemap,
// 	const OpnodeT* root)
// {
// 	for (auto child : root->children_)
// 	{
// 		auto& c = child.second;
// 		triemap.emplace(c, root);
// 		reverse_triemap(triemap, c);
// 	}
// }

// template <typename T>
// void leaf_triemap(
// 	std::unordered_map<eteq::Constant<T>*,OpnodesT>& cstmap,
// 	std::unordered_map<eteq::iLeaf<T>*,OpnodesT>& lefmap,
// 	const OpnodeT* root)
// {
// 	for (auto child : root->children_)
// 	{
// 		leaf_triemap(cstmap, lefmap, child.second);
// 	}
// 	if (root->leaf_.has_value())
// 	{
// 		for (const auto& leaf : root->leaf_.leaves_)
// 		{
// 			if (auto cst = dynamic_cast<eteq::Constant<T>*>(leaf.first))
// 			{
// 				cstmap[cst].emplace(root);
// 			}
// 			else
// 			{
// 				lefmap[static_cast<eteq::iLeaf<T>*>(leaf.first)].emplace(root);
// 			}
// 		}
// 	}
// }

void optimize (GraphInfo& graph, const OptRulesT& rules)
{
size_t i = 0;
	for (const OptRule& rule : rules)
	{
		query::Query q = graph.query_;
		rule.matcher_(q);
		query::QResultsT results;
		q.exec(results);
		for (query::QueryResult& result : results)
		{
std::cout << "converting " << i << ":";
PrettyEquation().print(std::cout, result.root_);
for (auto& s : result.symbs_)
{
std::cout << s.first << ":";
PrettyEquation().print(std::cout, s.second);
std::cout << std::endl;
}
		}
std::cout << std::endl;
++i;
	}
}

}

#endif
