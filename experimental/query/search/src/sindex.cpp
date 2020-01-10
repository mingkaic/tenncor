#include "experimental/query/search/sindex.hpp"

#ifdef SEARCH_SINDEX_HPP

namespace query
{

namespace search
{

using PathListT = std::list<PathNode>;

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
		auto children = func.get_children();
		PathInfoT& finfo = paths_[&func];
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto child = children[i];
			child->accept(*this);
			if (estd::has(paths_, child.get()))
			{
				// child is a functor with path info
				auto& cinfo = paths_[child.get()];
				for (auto pathpair : cinfo)
				{
					for (auto& path : pathpair.second)
					{
						path.push_front(PathNode{i, fop});
					}
					finfo[pathpair.first] = pathpair.second;
				}
			}
			else
			{
				// child is a leaf
				auto cleaf = static_cast<teq::iLeaf*>(child.get());
				finfo[cleaf].push_back(PathListT{PathNode{i, fop}});
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

}

}

#endif
