#include "query/sindex.hpp"

#ifdef SEARCH_SINDEX_HPP

namespace query
{

namespace search
{

void populate_itable (OpTrieT& itable, const OpPathMapT& opmap)
{
	for (auto& pathpair : opmap)
	{
		const PathInfo& info = pathpair.second;
		for (auto& leafpair : info.leaves_)
		{
			auto leaf = leafpair.first;
			auto& paths = leafpair.second;
			for (auto& path : paths)
			{
				itable.emplace(PathNodesT(path.begin(), path.end()),
					PathVal()).leaves_[leaf].emplace(pathpair.first);
			}
		}
		for (auto& attrpair : info.attrs_)
		{
			auto attr = attrpair.first;
			auto& paths = attrpair.second;
			for (auto& path : paths)
			{
				itable.emplace(PathNodesT(path.begin(), path.end()),
					PathVal()).attrs_[attr].emplace(pathpair.first);
			}
		}
	}
}

void populate_itable (OpTrieT& itable, const teq::TensptrsT& roots)
{
	OpPathBuilder builder;
	for (auto root : roots)
	{
		root->accept(builder);
	}
	populate_itable(itable, builder.paths_);
}

static void possible_paths_helper (const PathCbF& cb,
	PathListT& buffer, const OpTrieT::NodeT* node)
{
	assert(nullptr != node);
	if (node->leaf_.has_value())
	{
		cb(buffer, *node->leaf_);
	}
	if (node->children_.empty())
	{
		return;
	}
	buffer.push_back(PathNode{});
	std::vector<std::pair<PathNode,const OpTrieT::NodeT*>> cpairs(
		node->children_.begin(), node->children_.end());
	std::sort(cpairs.begin(), cpairs.end(),
		[](std::pair<PathNode,const OpTrieT::NodeT*>& a,
			std::pair<PathNode,const OpTrieT::NodeT*>& b)
		{
			return a.first < b.first;
		});
	for (const auto& cpair : cpairs)
	{
		const OpTrieT::NodeT* next = cpair.second;
		buffer.back() = cpair.first;
		possible_paths_helper(cb, buffer, next);
	}
	buffer.pop_back();
}

void possible_paths (const PathCbF& cb,
	const OpTrieT::NodeT* node)
{
	if (node)
	{
		PathListT buf;
		possible_paths_helper(cb, buf, node);
	}
}

void possible_paths (const PathCbF& cb,
	const OpTrieT& itable, const PathNodesT& path)
{
	possible_paths(cb, itable.match_prefix(path));
}

}

}

#endif
