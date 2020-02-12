//
/// sindex.hpp
/// search
///
/// Purpose:
/// Define OPCODE path tries
///

#ifndef SEARCH_SINDEX_HPP
#define SEARCH_SINDEX_HPP

#include <list>

#include <boost/functional/hash.hpp>

#include "estd/triebig.hpp"

#include "teq/traveler.hpp"

#include "eigen/generated/opcode.hpp"

#include "query/position.hpp"

namespace query
{

struct PathNode final
{
	friend bool operator == (const PathNode& a, const PathNode& b)
	{
		return a.idx_ == b.idx_ && a.opname_ == b.opname_;
	}

	friend bool operator < (const PathNode& a, const PathNode& b)
	{
		if (a.opname_ < b.opname_)
		{
			return true;
		}
		return a.idx_ < b.idx_;
	}

	size_t idx_;

	std::string opname_;
};

using PathNodesT = std::vector<PathNode>;

namespace search
{

using PathListT = std::list<PathNode>;

struct PathNodeHasher final
{
	size_t operator() (const PathNode& node) const
	{
		size_t seed = egen::is_commutative(node.opname_) ? 0 : node.idx_;
		boost::hash_combine(seed, node.opname_);
		return seed;
	}
};

using LSetMapT = teq::LeafMapT<teq::TensSetT>;

struct FuncVal final
{
	teq::TensSetT roots_;

	TensPosition positions_;
};

using FSetMapT = teq::FuncMapT<FuncVal>;

// path root R to leaf L key maps to mapping different mappings
struct PathVal final
{
	// Maps reachable leaf L -> set of path roots R
	LSetMapT leaves_;

	// Maps reachable attributable functor F -> set of path roots R
	FSetMapT attrs_;

	// Maps reachable commutative functor F -> set of path roots R
	FSetMapT comms_;
};

// Rationality:
// # of Path Keys = # Leaves
// size of PathVal value per leaf is proportional to length of max Path
// Realistically, number of edges of an operational graph

using OpTrieT = estd::Trie<PathNodesT,
	estd::TrieBigNode<PathNode,PathVal,PathNodeHasher>>;

using PathListsT = std::vector<PathListT>;

struct PathInfo final
{
	teq::LeafMapT<PathListsT> leaves_;

	teq::FuncMapT<PathListsT> attrs_;

	teq::FuncMapT<PathListsT> comms_;
};

using PathInfoT = teq::LeafMapT<PathListsT>;

using OpPathMapT = std::unordered_map<teq::iTensor*,PathInfo>;

struct OpPathBuilder final : public teq::iTraveler
{
	/// Implementation of iTraveler
	void visit (teq::iLeaf& leaf) override
	{
		if (false == estd::has(paths_, &leaf))
		{
			paths_[&leaf].leaves_[&leaf].push_back(PathListT{});
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor& func) override
	{
		if (false == estd::has(paths_, &func))
		{
			std::string fop = func.get_opcode().name_;
			PathInfo& finfo = paths_[&func];
			auto children = func.get_children();
			if (egen::is_commutative(fop))
			{
				paths_[&func].comms_[&func].push_back(PathListT{});
				for (auto& child : children)
				{
					child->accept(*this);
				}
				return;
			}
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				PathNode node{i, fop};
				auto child = children[i];
				child->accept(*this);
				// child is a functor with path info
				PathInfo& cinfo = paths_[child.get()];
				for (auto pathpair : cinfo.leaves_)
				{
					for (PathListT& path : pathpair.second)
					{
						path.push_front(node);
					}
					PathListsT& lentries = finfo.leaves_[pathpair.first];
					lentries.insert(lentries.end(),
						pathpair.second.begin(), pathpair.second.end());
				}
				for (auto pathpair : cinfo.attrs_)
				{
					for (PathListT& path : pathpair.second)
					{
						path.push_front(node);
					}
					PathListsT& lentries = finfo.attrs_[pathpair.first];
					lentries.insert(lentries.end(),
						pathpair.second.begin(), pathpair.second.end());
				}
				for (auto compair : cinfo.comms_)
				{
					for (PathListT& path : compair.second)
					{
						path.push_front(node);
					}
					PathListsT& centries = finfo.comms_[compair.first];
					centries.insert(centries.end(),
						compair.second.begin(), compair.second.end());
				}
				if (func.ls_attrs().size() > 0)
				{
					finfo.attrs_[&func].push_back(PathListT{node});
				}
			}
		}
	}

	OpPathMapT paths_;
};

void populate_itable (OpTrieT& itable, const OpPathMapT& opmap);

/// Populate index table of graph structure
void populate_itable (OpTrieT& itable, const teq::TensptrsT& roots);

using PathCbF = std::function<void(const PathListT&,const PathVal&)>;

void possible_paths (const PathCbF& cb, const OpTrieT::NodeT* node);

void possible_paths (const PathCbF& cb,
	const OpTrieT& itable, const PathNodesT& path);

}

}

#endif // SEARCH_SINDEX_HPP
