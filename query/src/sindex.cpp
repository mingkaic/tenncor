#include "query/sindex.hpp"

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
	void visit_leaf (teq::iLeaf& leaf) override
	{
		paths_[&leaf].leaves_[&leaf].push_back(PathListT{});
	}

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
		PathInfo& info = pathpair.second;
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

teq::iTensor* walk (teq::iTensor* root, PathListT& path)
{
	for (auto& node : path)
	{
		root = static_cast<teq::iFunctor*>(root)->
			get_children().at(node.idx_).get();
	}
	return root;
}

}

}

#endif
