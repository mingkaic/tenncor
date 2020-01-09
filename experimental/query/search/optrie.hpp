#ifndef SEARCH_OPTRIE_HPP
#define SEARCH_OPTRIE_HPP

#include <list>

#include <boost/functional/hash.hpp>

#include "teq/ifunctor.hpp"
#include "teq/itraveler.hpp"

#include "eigen/generated/opcode.hpp"

#include "experimental/query/search/trie.hpp"

namespace query
{

struct PathNode final
{
    size_t idx_;

    egen::_GENERATED_OPCODE op_;
};

using PathNodesT = std::vector<PathNode>;

namespace search
{

struct FuncVal final
{
    teq::LeafSetT leaves_;

    teq::FuncSetT funcs_;
};

struct PathHasher final
{
    size_t operator() (const PathNode& node) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, node.op_);
        boost::hash_combine(seed, node.idx_);
        return seed;
    }
};

using OpTrieT = estd::Trie<PathNodesT,FuncVal,PathHasher>;

struct OpPath final
{
    egen::_GENERATED_OPCODE opcode_;

    std::vector<std::shared<OpPath>> nexts_;
};

using OpPathptrT = std::shared<OpPath>;

struct OpPathBuilder final : public teq::iOnceTraveler
{
    std::unordered_map<teq::iTensor*,OpPathptrT> roots_;

private:
    void visit_leaf (teq::iLeaf& leaf) override {}

    void visit_func (teq::iFunctor& func) override
    {
        auto opnode = std::make_shared<OpPath>();
        opnode->opcode_ = func->get_opcode().code_;
        auto children = func.get_children();
        for (auto child : children)
        {
            child->accept(*this);
            opnode->nexts_.push_back(estd::try_get(
                roots_, child.get(), nullptr));
        }
        roots_.emplace(&func, opnode);
    }
};

void populate_table (OpTrieT& table, teq::TensptrsT roots)
{
	teq::OwnerMapT owners = teq::track_owners(roots);
    OpPathBuilder builder;
    for (auto root : roots)
    {
        root->accept(builder);
    }
}

}

}

#endif // SEARCH_OPTRIE_HPP
