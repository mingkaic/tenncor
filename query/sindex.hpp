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

namespace query
{

struct PathNode final
{
	friend bool operator == (const PathNode& a, const PathNode& b)
	{
		bool codeq = a.op_ == b.op_;
		// todo: stop paths at commutative operators
		if (egen::is_commutative(a.op_) && codeq)
		{
			// ignore idx
			return true;
		}
		return a.idx_ == b.idx_ && codeq;
	}

	friend bool operator < (const PathNode& a, const PathNode& b)
	{
		if (egen::name_op(a.op_) < egen::name_op(b.op_))
		{
			return true;
		}
		return a.idx_ < b.idx_;
	}

	size_t idx_;

	egen::_GENERATED_OPCODE op_;
};

using PathNodesT = std::vector<PathNode>;

namespace search
{

using PathListT = std::list<PathNode>;

struct PathNodeHasher final
{
	size_t operator() (const PathNode& node) const
	{
		size_t seed = egen::is_commutative(node.op_) ? 0 : node.idx_;
		boost::hash_combine(seed, node.op_);
		return seed;
	}
};

using LSetMapT = teq::LeafMapT<teq::TensSetT>;

using FSetMapT = teq::FuncMapT<teq::TensSetT>;

// path root R to leaf L key maps to mapping different mappings
struct PathVal final
{
	// Maps reachable leaf L -> set of path roots R
	LSetMapT leaves_;

	// Maps reachable attributable functor F -> set of path roots R
	FSetMapT attrs_;
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
