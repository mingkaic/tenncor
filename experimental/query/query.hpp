#ifndef QUERY_HPP
#define QUERY_HPP

#include "experimental/query/search/sindex.hpp"

#include "experimental/query/parse.hpp"

namespace query
{

struct Query
{
    Query (search::OpTrieT& sindex) : sindex_(sindex) {}

    void where (std::istream& condition,
        size_t depth = std::numerical_limits<size_t>::max()
    {
        Node cond;
        json_parse(cond, condition);

        teq::TensSetT results;
        constraint(results, &cond);
    }

private:
    void constraint (teq::TensSetT& results, const Node* cond)
    {
        // todo: pass path
        switch (cond->node_case())
        {
            case Node::NodeCase::kCst:
            {
                double scalar = cond.cst();
                // todo: verify leaves_
            }
                break;
            case Node::NodeCase::kVar:
            {
                const Variable& var = cond.var();
                // todo: verify leaves_
            }
                break;
            case Node::NodeCase::kOp:
            {
                Operator* op = cond.op();
                // todo: verify attrs_
                const auto& args = op->args();
                for (const Node* arg : args)
                {
                    constraint(results, arg);
                }
            }
                break;
            default:
                logs::fatal("cannot process unset graph node");
        }
    }

    search::OpTrieT& sindex_;
};

}

#endif // QUERY_HPP
