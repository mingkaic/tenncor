#include "experimental/query/search/sindex.hpp"

#ifdef SEARCH_SINDEX_HPP

namespace query
{

namespace search
{

using PathListsT = std::vector<PathListT>;

using PathInfoT = teq::LeafMapT<PathListsT>;

struct OpPathBuilder final : public teq::iOnceTraveler
{
	std::unordered_map<teq::iTensor*,PathInfoT> paths_;

private:
	void visit_leaf (teq::iLeaf& leaf) override {}

	void visit_func (teq::iFunctor& func) override
	{
		egen::_GENERATED_OPCODE fop =
			(egen::_GENERATED_OPCODE) func.get_opcode().code_;
		bool is_comm = egen::is_commutative(fop);
		auto children = func.get_children();
		PathInfoT& finfo = paths_[&func];
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto child = children[i];
			child->accept(*this);
			if (estd::has(paths_, child.get()))
			{
				// child is a functor with path info
				PathInfoT& cinfo = paths_[child.get()];
				for (std::pair<teq::iLeaf*,PathListsT> pathpair : cinfo)
				{
					for (PathListT& path : pathpair.second)
					{
						path.push_front(PathNode{is_comm ? 0 : i, fop});
					}
					PathListsT& lentries = finfo[pathpair.first];
					lentries.insert(lentries.end(),
						pathpair.second.begin(), pathpair.second.end());
				}
			}
			else
			{
				// child is a leaf
				auto cleaf = static_cast<teq::iLeaf*>(child.get());
				finfo[cleaf].push_back(PathListT{PathNode{is_comm ? 0 : i, fop}});
			}
		}
	}
};

void populate_itable (OpTrieT& itable, teq::TensptrsT roots)
{
	teq::OwnerMapT owners = teq::track_owners(roots);
	OpPathBuilder builder;
	for (auto root : roots)
	{
		root->accept(builder);
	}
	for (auto& pathpair : builder.paths_)
	{
		auto func = static_cast<teq::iFunctor*>(pathpair.first);
		PathInfoT& info = pathpair.second;
		for (auto& leafpair : info)
		{
			auto leaf = leafpair.first;
			auto& paths = leafpair.second;
			for (auto& path : paths)
			{
				itable.emplace(PathNodesT(path.begin(), path.end()),
					PathVal())[leaf].emplace(func);
			}
		}
	}
}

static void possible_paths_helper (PathCbF& cb,
	PathListT& buffer, const OpTrieT::TrieNodeT* node)
{
	if (node->leaf_.has_value())
	{
		cb(buffer, *node->leaf_);
	}
	if (node->children_.empty())
	{
		return;
	}
	buffer.push_back(PathNode{});
	std::vector<std::pair<PathNode,const OpTrieT::TrieNodeT*>> cpairs(
		node->children_.begin(), node->children_.end());
	std::sort(cpairs.begin(), cpairs.end(),
		[](std::pair<PathNode,const OpTrieT::TrieNodeT*>& a,
			std::pair<PathNode,const OpTrieT::TrieNodeT*>& b)
		{
			return a.first < b.first;
		});
	for (const auto& cpair : cpairs)
	{
		const OpTrieT::TrieNodeT* next = cpair.second;
		buffer.back() = cpair.first;
		possible_paths_helper(cb, buffer, next);
	}
	buffer.pop_back();
}

void possible_paths (PathCbF& cb,
	const OpTrieT& itable, const PathNodesT& path)
{
	PathListT buf;
	possible_paths_helper(cb, buf, itable.match_prefix(path));
}

}

}

#endif
