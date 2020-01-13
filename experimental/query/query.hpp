#ifndef QUERY_HPP
#define QUERY_HPP

#include "experimental/query/search/sindex.hpp"

#include "experimental/query/parse.hpp"

namespace query
{

static inline teq::Shape to_shape (
    const google::protobuf::RepeatedField<uint32_t>& sfields)
{
    std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
    return teq::Shape(slist);
}

static inline bool equals (double scalar, const teq::iLeaf* leaf)
{
    return teq::IMMUTABLE == leaf->get_usage() &&
        leaf->to_string() == fmts::to_string(scalar);
}

static inline bool equals (const Variable& var, const teq::iLeaf* leaf)
{
    return (false == var.has_label() || var.label() == leaf->to_string()) &&
        (false == var.has_dtype() || var.dtype() == leaf->type_label()) &&
        (0 == var.shape_size() || to_shape(var.shape()).compatible_after(leaf->shape(), 0));
}

struct Query
{
    Query (search::OpTrieT& sindex) : sindex_(sindex) {}

    void where (teq::TensSetT& results, std::istream& condition)
    {
        Node cond;
        json_parse(cond, condition);

        const search::OpTrieT::TrieNodeT* tri = sindex_.root();
        constraint(results, tri, cond);
    }

private:
    bool equals (const Attribute& pba, const marsh::iObject* attr)
    {
        bool match = false;
        switch (pba.type())
        {
            case Attribute::INT:
                if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
                {
                    match = pba.inum() == num->to_int64();
                }
                break;
            case Attribute::DOUBLE:
                if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
                {
                    match = pba.dnum() == num->to_float64();
                }
                break;
            case Attribute::INT_ARR:
                if (auto narr = dynamic_cast<const marsh::iArray*>(attr))
                {
                    const auto& arr = pba.iarr();
                    if (arr.size() == narr->size())
                    {
                        match = true;
                        narr->foreach(
                        [&](size_t i, const marsh::ObjptrT& obj)
                        {
                            auto num = dynamic_cast<const marsh::iNumber*>(obj.get());
                            match = match &&
                                nullptr != num && arr[i] == num->to_int64();
                        });
                    }
                }
                break;
            case Attribute::DOUBLE_ARR:
                if (auto narr = dynamic_cast<const marsh::iArray*>(attr))
                {
                    const auto& arr = pba.darr();
                    if (arr.size() == narr->size())
                    {
                        match = true;
                        narr->foreach(
                        [&](size_t i, const marsh::ObjptrT& obj)
                        {
                            auto num = dynamic_cast<const marsh::iNumber*>(obj.get());
                            match = match &&
                                nullptr != num && arr[i] == num->to_float64();
                        });
                    }
                }
                break;
            case Attribute::STRING:
                match = pba.str() == attr->to_string();
                break;
            case Attribute::NODE:
            {
                if (auto tens = dynamic_cast<const teq::TensorObj*>(attr))
                {
                    teq::TensSetT results;
                    constraint(results, sindex_.root(), pba.node());
                    match = estd::has(results, tens->get_tensor().get());
                }
            }
                break;
            case Attribute::LAYER:
            {
                if (auto lay = dynamic_cast<const teq::LayerObj*>(attr))
                {
                    const Layer& layer = pba.layer();
                    if (layer.name() == lay->get_opname())
                    {
                        teq::TensSetT results;
                        constraint(results, sindex_.root(), layer.input());
                        match = estd::has(results, lay->get_tensor().get());
                    }
                }
            }
                break;
            default:
                logs::fatal("cannot compare unknown attribute");
        }
        return match;
    }

    // Return true if there's at least one result
    void constraint (teq::TensSetT& results,
        const search::OpTrieT::TrieNodeT* tri, const Node& cond) // todo: add filter_out mechanism instead of allocing a subresult at every branch
    {
        if (nullptr == tri)
        {
            results.clear();
            return;
        }
        switch (cond.type())
        {
            case Node::CONSTANT:
            {
                if (false == tri->leaf_.has_value())
                {
                    results.clear();
                    return;
                }
                bool nomatch = true;
                double scalar = cond.cst();
                for (const auto& lpair : tri->leaf_->leaves_)
                {
                    if (::query::equals(scalar, lpair.first))
                    {
                        nomatch = false;
                        results.insert(lpair.second.begin(), lpair.second.end());
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
                const Variable& var = cond.var();
                for (const auto& lpair : tri->leaf_->leaves_)
                {
                    if (::query::equals(var, lpair.first))
                    {
                        nomatch = false;
                        results.insert(lpair.second.begin(), lpair.second.end());
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
                const Operator& op = cond.op();
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
                    teq::TensSetT subresults;
                    iterate_condition(subresults, sindex_.root(), op);
                    // get attr_matches[subresults intersection attr_matches.keys]
                    if (subresults.empty())
                    {
                        results.clear();
                    }
                    for (const auto& apair : attr_matches)
                    {
                        if (estd::has(subresults, apair.first))
                        {
                            results.insert(apair.second.begin(), apair.second.end());
                        }
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
    void iterate_condition (teq::TensSetT& results,
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
                        results.insert(lpair.second.begin(), lpair.second.end());
                    }
                    for (const auto& apair : val.attrs_)
                    {
                        results.insert(apair.second.begin(), apair.second.end());
                    }
                }, tri);
            return;
        }
        constraint(results, search::OpTrieT::next(
            tri, PathNode{0, opcode}), args[0]);
        teq::TensSetT subresults;
        for (size_t i = 1, n = args.size(); i < n && false == results.empty(); ++i)
        {
            constraint(subresults, search::OpTrieT::next(
                tri, PathNode{i, opcode}), args[i]);
            // final result is an intersect of all subresults
            for (auto it = results.begin(), et = results.end();
                it != et;)
            {
                if (false == estd::has(subresults, *it))
                {
                    it = results.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            subresults.clear();
        }
    }

    search::OpTrieT& sindex_;
};

}

#endif // QUERY_HPP
