#include "experimental/query/search/sindex.hpp"

#ifdef SEARCH_SINDEX_HPP

namespace query
{

namespace search
{

using PathListsT = std::vector<PathListT>;

struct PathInfo final
{
	teq::LeafMapT<PathListsT> leaves_;

	teq::FuncMapT<PathListsT> attrs_;
};

using PathInfoT = teq::LeafMapT<PathListsT>;

struct OpPathBuilder final : public teq::iOnceTraveler
{
	std::unordered_map<teq::iTensor*,PathInfo> paths_;

private:
	void visit_leaf (teq::iLeaf& leaf) override {}

	void visit_func (teq::iFunctor& func) override
	{
		egen::_GENERATED_OPCODE fop =
			(egen::_GENERATED_OPCODE) func.get_opcode().code_;
		bool is_comm = egen::is_commutative(fop);
		auto children = func.get_children();
		PathInfo& finfo = paths_[&func];
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			PathNode node{is_comm ? 0 : i, fop};
			auto child = children[i];
			child->accept(*this);
			if (estd::has(paths_, child.get()))
			{
				// child is a functor with path info
				PathInfo& cinfo = paths_[child.get()];
				for (std::pair<teq::iLeaf*,PathListsT>
					pathpair : cinfo.leaves_)
				{
					for (PathListT& path : pathpair.second)
					{
						path.push_front(node);
					}
					PathListsT& lentries = finfo.leaves_[pathpair.first];
					lentries.insert(lentries.end(),
						pathpair.second.begin(), pathpair.second.end());
				}
				for (std::pair<teq::iFunctor*,PathListsT>
					pathpair : cinfo.attrs_)
				{
					for (PathListT& path : pathpair.second)
					{
						path.push_front(node);
					}
					PathListsT& lentries = finfo.attrs_[pathpair.first];
					lentries.insert(lentries.end(),
						pathpair.second.begin(), pathpair.second.end());
				}
				if (func.ls_attrs().size() > 0)
				{
					finfo.attrs_[&func].push_back(PathListT{node});
				}
			}
			else
			{
				// child is a leaf
				auto cleaf = static_cast<teq::iLeaf*>(child.get());
				finfo.leaves_[cleaf].push_back(PathListT{node});
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
		PathInfo& info = pathpair.second;
		for (auto& leafpair : info.leaves_)
		{
			auto leaf = leafpair.first;
			auto& paths = leafpair.second;
			for (auto& path : paths)
			{
				itable.emplace(PathNodesT(path.begin(), path.end()),
					PathVal()).leaves_[leaf].emplace(func);
			}
		}
		for (auto& attrpair : info.attrs_)
		{
			auto attr = attrpair.first;
			auto& paths = attrpair.second;
			for (auto& path : paths)
			{
				itable.emplace(PathNodesT(path.begin(), path.end()),
					PathVal()).attrs_.emplace(attr);
			}
		}
	}
}

static void possible_paths_helper (const PathCbF& cb,
	PathListT& buffer, const OpTrieT::TrieNodeT* node)
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

void possible_paths (const PathCbF& cb,
	const OpTrieT& itable, const PathNodesT& path)
{
	if (const OpTrieT::TrieNodeT* next = itable.match_prefix(path))
	{
		PathListT buf;
		possible_paths_helper(cb, buf, next);
	}
}

}

}

#endif
