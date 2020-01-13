#ifndef QUERY_HPP
#define QUERY_HPP

#include "experimental/query/search/sindex.hpp"

#include "experimental/query/parse.hpp"

namespace query
{

static inline bool equals (double scalar, const teq::iLeaf* leaf)
{
    return teq::IMMUTABLE == leaf->get_usage() &&
        leaf->to_string() == fmts::to_string(scalar);
}

static inline bool equals (const Variable& var, const teq::iLeaf* leaf)
{
    //
    return false;
}

static inline bool equals (const Attribute& pba, const marsh::iObject* attr)
{
    //
    return false;
}

struct WhereResults final
{
    // unionize roots
    void merge_roots (const teq::TensSetT& roots)
    {
        known_roots_.insert(roots.begin(), roots.end());
    }

    void apply_filter (const WhereResults& filter)
    {
        // final result is an intersect of all subresults
        for (auto it = known_roots_.begin(), et = known_roots_.end();
            it != et;)
        {
            if (false == filter.has(*it))
            {
                it = known_roots_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void clear (void)
    {
        known_roots_.clear();
    }

    bool empty (void) const
    {
        return known_roots_.empty();
    }

    bool has (teq::iTensor* const root) const
    {
        return estd::has(known_roots_, root);
    }

    teq::TensSetT known_roots_;
};

struct Query
{
    Query (search::OpTrieT& sindex) : sindex_(sindex) {}

    void where (teq::TensSetT& results, std::istream& condition)
    {
        Node cond;
        json_parse(cond, condition);

        WhereResults where;
        const search::OpTrieT::TrieNodeT* tri = sindex_.root();
        constraint(where, tri, &cond);
        results = where.known_roots_;
    }

private:
    // Return true if there's at least one result
    void constraint (WhereResults& results,
        const search::OpTrieT::TrieNodeT* tri, const Node* cond) // todo: add filter_out mechanism instead of allocing a subresult at every branch
    {
        if (nullptr == tri)
        {
            results.clear();
            return;
        }
        switch (cond->type())
        {
            case Node::CONSTANT:
            {
                if (false == tri->leaf_.has_value())
                {
                    results.clear();
                    return;
                }
                bool nomatch = true;
                double scalar = cond->cst();
                for (const auto& lpair : tri->leaf_->leaves_)
                {
                    if (equals(scalar, lpair.first))
                    {
                        nomatch = false;
                        results.merge_roots(lpair.second);
                    }
                }
                if (nomatch)
                {
                    results.clear();
                }
            }
                break;
            case Node::VARIABLE:
            {
                if (false == tri->leaf_.has_value())
                {
                    results.clear();
                    return;
                }
                bool nomatch = true;
                const Variable& var = cond->var();
                for (const auto& lpair : tri->leaf_->leaves_)
                {
                    if (equals(var, lpair.first))
                    {
                        nomatch = false;
                        results.merge_roots(lpair.second);
                    }
                }
                if (nomatch)
                {
                    results.clear();
                }
            }
                break;
            case Node::OPERATOR:
            {
                const Operator& op = cond->op();
                const auto& attrs = op.attrs();
                if (attrs.size() > 0)
                {
                    if (false == tri->leaf_.has_value() ||
                        tri->leaf_->attrs_.empty())
                    {
                        results.clear();
                        return;
                    }
                    teq::FuncMapT<teq::TensSetT> attr_matches;
                    for (const auto& apair : tri->leaf_->attrs_)
                    {
                        teq::iFunctor* iattr = apair.first;
                        std::unordered_set<std::string> need_keys;
                        for (const auto& apair : attrs)
                        {
                            need_keys.emplace(apair.first);
                        }
                        for (auto jt = need_keys.begin(), et = need_keys.end();
                            jt != et;)
                        {
                            std::string key = *jt;
                            auto val = iattr->get_attr(key);
                            if (nullptr != val && equals(attrs.at(key), val))
                            {
                                jt = need_keys.erase(jt);
                            }
                            else
                            {
                                ++jt;
                            }
                        }
                        if (need_keys.empty())
                        {
                            attr_matches.emplace(iattr, apair.second);
                        }
                    }
                    // match the rest of the condition subgraph from trie root
                    // in order to filter for matching attributable functors
                    WhereResults subresults;
                    iterate_condition(subresults, sindex_.root(), op);
                    // get attr_matches[subresults intersection attr_matches.keys]
                    for (const auto& apair : attr_matches)
                    {
                        if (subresults.has(apair.first))
                        {
                            results.merge_roots(apair.second);
                        }
                    }
                    if (subresults.empty())
                    {
                        results.clear();
                    }
                }
                else
                {
                    iterate_condition(results, tri, op);
                }
            }
                break;
            default:
                logs::fatal("cannot process unset graph node");
        }
    }

    // Return true if there's at least one result
    void iterate_condition (WhereResults& results,
        const search::OpTrieT::TrieNodeT* tri, const Operator& op)
    {
        egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
        const auto& args = op.args();
        if (args.empty())
        {
            search::possible_paths(
                [&](const search::PathListT& path, const search::PathVal& val)
                {
                    for (const auto& lpair : val.leaves_)
                    {
                        results.merge_roots(lpair.second);
                    }
                    for (const auto& apair : val.attrs_)
                    {
                        results.merge_roots(apair.second);
                    }
                }, tri);
            return;
        }
        constraint(results, search::OpTrieT::next(
            tri, PathNode{0, opcode}), &args[0]);
        WhereResults subresults;
        for (size_t i = 1, n = args.size(); i < n && false == results.empty(); ++i)
        {
            constraint(subresults, search::OpTrieT::next(
                tri, PathNode{i, opcode}), &args[i]);
            results.apply_filter(subresults);
            subresults.clear();
        }
    }

    search::OpTrieT& sindex_;
};

}

#endif // QUERY_HPP
