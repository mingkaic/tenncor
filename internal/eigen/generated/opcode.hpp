#include <string>
#include "logs/logs.hpp"
#include "internal/eigen/operator.hpp"
#include "internal/eigen/packattr.hpp"


#ifndef _GENERATED_OPCODES_HPP
#define _GENERATED_OPCODES_HPP

namespace egen
{

enum _GENERATED_OPCODE
{
    BAD_OP = 0,
    //>>> opcodes
    IDENTITY,
    ABS,
    NEG,
    SIN,
    COS,
    TAN,
    EXP,
    LOG,
    SQRT,
    ROUND,
    SIGMOID,
    TANH,
    SQUARE,
    CUBE,
    RAND_UNIF,
    REVERSE,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    ARGMAX,
    PERMUTE,
    EXTEND,
    RESHAPE,
    SLICE,
    PAD,
    STRIDE,
    SCATTER,
    POW,
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    EQ,
    NEQ,
    LT,
    GT,
    MATMUL,
    CONV,
    SELECT,
    CONCAT,
    ASSIGN,
    ASSIGN_ADD,
    ASSIGN_SUB,
    ASSIGN_MUL,
    ASSIGN_DIV,
    CAST,
    _N_GENERATED_OPCODES,
};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (const std::string& name);

bool is_commutative (_GENERATED_OPCODE code);

bool is_commutative (const std::string& name);

bool is_idempotent (_GENERATED_OPCODE code);

bool is_idempotent (const std::string& name);

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode, eigen::EigenptrT& out, teq::Shape outshape, const teq::TensptrsT& in, const marsh::iAttributed& attrib)
{
    //>>> ^ params
    switch (opcode)
    {
        //>>> ops
        case IDENTITY: out = eigen::ref<T>(outshape, *in[0]); break;
        case ABS: out = eigen::abs<T>(outshape, *in[0]); break;
        case NEG: out = eigen::neg<T>(outshape, *in[0]); break;
        case SIN: out = eigen::sin<T>(outshape, *in[0]); break;
        case COS: out = eigen::cos<T>(outshape, *in[0]); break;
        case TAN: out = eigen::tan<T>(outshape, *in[0]); break;
        case EXP: out = eigen::exp<T>(outshape, *in[0]); break;
        case LOG: out = eigen::log<T>(outshape, *in[0]); break;
        case SQRT: out = eigen::sqrt<T>(outshape, *in[0]); break;
        case ROUND: out = eigen::round<T>(outshape, *in[0]); break;
        case SIGMOID: out = eigen::sigmoid<T>(outshape, *in[0]); break;
        case TANH: out = eigen::tanh<T>(outshape, *in[0]); break;
        case SQUARE: out = eigen::square<T>(outshape, *in[0]); break;
        case CUBE: out = eigen::cube<T>(outshape, *in[0]); break;
        case RAND_UNIF: out = eigen::rand_uniform<T>(outshape, *in[0], *in[1]); break;
        case REVERSE: out = eigen::reverse<T>(outshape, *in[0], attrib); break;
        case REDUCE_SUM: out = eigen::reduce_sum<T>(outshape, *in[0], attrib); break;
        case REDUCE_PROD: out = eigen::reduce_prod<T>(outshape, *in[0], attrib); break;
        case REDUCE_MIN: out = eigen::reduce_min<T>(outshape, *in[0], attrib); break;
        case REDUCE_MAX: out = eigen::reduce_max<T>(outshape, *in[0], attrib); break;
        case ARGMAX: out = eigen::argmax<T>(outshape, *in[0], attrib); break;
        case PERMUTE: out = eigen::permute<T>(outshape, *in[0], attrib); break;
        case EXTEND: out = eigen::extend<T>(outshape, *in[0], attrib); break;
        case RESHAPE: out = eigen::reshape<T>(outshape, *in[0]); break;
        case SLICE: out = eigen::slice<T>(outshape, *in[0], attrib); break;
        case PAD: out = eigen::pad<T>(outshape, *in[0], attrib); break;
        case STRIDE: out = eigen::stride<T>(outshape, *in[0], attrib); break;
        case SCATTER: out = eigen::scatter<T>(outshape, *in[0], attrib); break;
        case POW: out = eigen::pow<T>(outshape, *in[0], *in[1]); break;
        case ADD: out = eigen::add<T>(outshape, in); break;
        case SUB: out = eigen::sub<T>(outshape, *in[0], *in[1]); break;
        case MUL: out = eigen::mul<T>(outshape, in); break;
        case DIV: out = eigen::div<T>(outshape, *in[0], *in[1]); break;
        case MIN: out = eigen::min<T>(outshape, *in[0], *in[1]); break;
        case MAX: out = eigen::max<T>(outshape, *in[0], *in[1]); break;
        case EQ: out = eigen::eq<T>(outshape, *in[0], *in[1]); break;
        case NEQ: out = eigen::neq<T>(outshape, *in[0], *in[1]); break;
        case LT: out = eigen::lt<T>(outshape, *in[0], *in[1]); break;
        case GT: out = eigen::gt<T>(outshape, *in[0], *in[1]); break;
        case MATMUL: out = eigen::matmul<T>(outshape, *in[0], *in[1], attrib); break;
        case CONV: out = eigen::convolution<T>(outshape, *in[0], *in[1], attrib); break;
        case SELECT: out = eigen::select<T>(outshape, *in[0], *in[1], *in[2]); break;
        case CONCAT: out = eigen::concat<T>(outshape, in, attrib); break;
        case ASSIGN: out = eigen::assign<T>(*in[0], *in[1]); break;
        case ASSIGN_ADD: out = eigen::assign_add<T>(*in[0], *in[1]); break;
        case ASSIGN_SUB: out = eigen::assign_sub<T>(*in[0], *in[1]); break;
        case ASSIGN_MUL: out = eigen::assign_mul<T>(*in[0], *in[1]); break;
        case ASSIGN_DIV: out = eigen::assign_div<T>(*in[0], *in[1]); break;
        case CAST: out = eigen::cast<T>(*in[0]); break;
        default: global::fatal("unknown opcode");
    }
}

// GENERIC_MACRO must accept a static opcode as an argument.
// e.g.:
// #define GENERIC_MACRO(COMPILE_OPCODE) run<COMPILE_OPCODE>(args...);
// ...
// OPCODE_LOOKUP(GENERIC_MACRO, rt_opcode)
// this is used for mapping compile-time ops using runtime opcode variable
#define OPCODE_LOOKUP(GENERIC_MACRO, OPCODE)\
switch (OPCODE)\
{\
    case egen::IDENTITY: GENERIC_MACRO(::egen::IDENTITY) break;\
    case egen::ABS: GENERIC_MACRO(::egen::ABS) break;\
    case egen::NEG: GENERIC_MACRO(::egen::NEG) break;\
    case egen::SIN: GENERIC_MACRO(::egen::SIN) break;\
    case egen::COS: GENERIC_MACRO(::egen::COS) break;\
    case egen::TAN: GENERIC_MACRO(::egen::TAN) break;\
    case egen::EXP: GENERIC_MACRO(::egen::EXP) break;\
    case egen::LOG: GENERIC_MACRO(::egen::LOG) break;\
    case egen::SQRT: GENERIC_MACRO(::egen::SQRT) break;\
    case egen::ROUND: GENERIC_MACRO(::egen::ROUND) break;\
    case egen::SIGMOID: GENERIC_MACRO(::egen::SIGMOID) break;\
    case egen::TANH: GENERIC_MACRO(::egen::TANH) break;\
    case egen::SQUARE: GENERIC_MACRO(::egen::SQUARE) break;\
    case egen::CUBE: GENERIC_MACRO(::egen::CUBE) break;\
    case egen::RAND_UNIF: GENERIC_MACRO(::egen::RAND_UNIF) break;\
    case egen::REVERSE: GENERIC_MACRO(::egen::REVERSE) break;\
    case egen::REDUCE_SUM: GENERIC_MACRO(::egen::REDUCE_SUM) break;\
    case egen::REDUCE_PROD: GENERIC_MACRO(::egen::REDUCE_PROD) break;\
    case egen::REDUCE_MIN: GENERIC_MACRO(::egen::REDUCE_MIN) break;\
    case egen::REDUCE_MAX: GENERIC_MACRO(::egen::REDUCE_MAX) break;\
    case egen::ARGMAX: GENERIC_MACRO(::egen::ARGMAX) break;\
    case egen::PERMUTE: GENERIC_MACRO(::egen::PERMUTE) break;\
    case egen::EXTEND: GENERIC_MACRO(::egen::EXTEND) break;\
    case egen::RESHAPE: GENERIC_MACRO(::egen::RESHAPE) break;\
    case egen::SLICE: GENERIC_MACRO(::egen::SLICE) break;\
    case egen::PAD: GENERIC_MACRO(::egen::PAD) break;\
    case egen::STRIDE: GENERIC_MACRO(::egen::STRIDE) break;\
    case egen::SCATTER: GENERIC_MACRO(::egen::SCATTER) break;\
    case egen::POW: GENERIC_MACRO(::egen::POW) break;\
    case egen::ADD: GENERIC_MACRO(::egen::ADD) break;\
    case egen::SUB: GENERIC_MACRO(::egen::SUB) break;\
    case egen::MUL: GENERIC_MACRO(::egen::MUL) break;\
    case egen::DIV: GENERIC_MACRO(::egen::DIV) break;\
    case egen::MIN: GENERIC_MACRO(::egen::MIN) break;\
    case egen::MAX: GENERIC_MACRO(::egen::MAX) break;\
    case egen::EQ: GENERIC_MACRO(::egen::EQ) break;\
    case egen::NEQ: GENERIC_MACRO(::egen::NEQ) break;\
    case egen::LT: GENERIC_MACRO(::egen::LT) break;\
    case egen::GT: GENERIC_MACRO(::egen::GT) break;\
    case egen::MATMUL: GENERIC_MACRO(::egen::MATMUL) break;\
    case egen::CONV: GENERIC_MACRO(::egen::CONV) break;\
    case egen::SELECT: GENERIC_MACRO(::egen::SELECT) break;\
    case egen::CONCAT: GENERIC_MACRO(::egen::CONCAT) break;\
    case egen::ASSIGN: GENERIC_MACRO(::egen::ASSIGN) break;\
    case egen::ASSIGN_ADD: GENERIC_MACRO(::egen::ASSIGN_ADD) break;\
    case egen::ASSIGN_SUB: GENERIC_MACRO(::egen::ASSIGN_SUB) break;\
    case egen::ASSIGN_MUL: GENERIC_MACRO(::egen::ASSIGN_MUL) break;\
    case egen::ASSIGN_DIV: GENERIC_MACRO(::egen::ASSIGN_DIV) break;\
    case egen::CAST: GENERIC_MACRO(::egen::CAST) break;\
    default: global::fatal("executing bad op");\
}
//>>> ^ cases

//>>> per_op

template <_GENERATED_OPCODE OPCODE>
struct FuncOpt final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        return false;
    }
};


template <_GENERATED_OPCODE OPCODE>
struct ShapeParser final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        teq::Shape outshape = shapes.front();
        for (size_t i = 1, n = shapes.size(); i < n; ++i)
        {
            if (false == shapes[i].compatible_after(outshape, 0))
            {
                global::throw_errf("cannot %s with incompatible shapes %s and %s",
                    egen::name_op(OPCODE).c_str(),
                    shapes[i].to_string().c_str(),
                    outshape.to_string().c_str());
            }
        }
        return outshape;

    }
};


template <_GENERATED_OPCODE OPCODE>
struct TypeParser final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return *std::max_element(dtypes.begin(), dtypes.end(),
        [](egen::_GENERATED_DTYPE lhs, egen::_GENERATED_DTYPE rhs)
        {
            return egen::type_precision(lhs) < egen::type_precision(rhs);
        });

    }
};


template <>
struct ShapeParser<IDENTITY> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return shapes.front();

    }
};


template <>
struct FuncOpt<REDUCE_SUM> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
    std::set<teq::RankT> ranks;
    eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
    bool redundant = ranks.empty();
    if (redundant)
    {
        global::debugf("reducing with no significant dimensions... "
            "treating as identity: (dims=%s, shape=%s)",
            fmts::to_string(ranks.begin(), ranks.end()).c_str(),
            args.front()->shape().to_string().c_str());
    }
    return redundant;

    }
};


template <>
struct ShapeParser<REDUCE_SUM> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        std::set<teq::RankT> ranks;
        eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (teq::RankT i : ranks)
        {
            slist[i] = 1;
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<REDUCE_PROD> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
    std::set<teq::RankT> ranks;
    eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
    bool redundant = ranks.empty();
    if (redundant)
    {
        global::debugf("reducing with no significant dimensions... "
            "treating as identity: (dims=%s, shape=%s)",
            fmts::to_string(ranks.begin(), ranks.end()).c_str(),
            args.front()->shape().to_string().c_str());
    }
    return redundant;

    }
};


template <>
struct ShapeParser<REDUCE_PROD> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        std::set<teq::RankT> ranks;
        eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (teq::RankT i : ranks)
        {
            slist[i] = 1;
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<REDUCE_MIN> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
    std::set<teq::RankT> ranks;
    eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
    bool redundant = ranks.empty();
    if (redundant)
    {
        global::debugf("reducing with no significant dimensions... "
            "treating as identity: (dims=%s, shape=%s)",
            fmts::to_string(ranks.begin(), ranks.end()).c_str(),
            args.front()->shape().to_string().c_str());
    }
    return redundant;

    }
};


template <>
struct ShapeParser<REDUCE_MIN> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        std::set<teq::RankT> ranks;
        eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (teq::RankT i : ranks)
        {
            slist[i] = 1;
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<REDUCE_MAX> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
    std::set<teq::RankT> ranks;
    eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
    bool redundant = ranks.empty();
    if (redundant)
    {
        global::debugf("reducing with no significant dimensions... "
            "treating as identity: (dims=%s, shape=%s)",
            fmts::to_string(ranks.begin(), ranks.end()).c_str(),
            args.front()->shape().to_string().c_str());
    }
    return redundant;

    }
};


template <>
struct ShapeParser<REDUCE_MAX> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        std::set<teq::RankT> ranks;
        eigen::Packer<std::set<teq::RankT>>().unpack(ranks, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (teq::RankT i : ranks)
        {
            slist[i] = 1;
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<ARGMAX> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        teq::RankT return_dim;
        eigen::Packer<teq::RankT>().unpack(return_dim, attrs);
        teq::Shape shape = args.front()->shape();
        bool redundant = return_dim < teq::rank_cap && shape.at(return_dim) == 1;
        if (redundant)
        {
            global::debugf("argreducing with no significant dimensions... "
                "treating as identity: (return_dim=%d, shape=%s)",
                (int) return_dim, shape.to_string().c_str());
        }
        return redundant;

    }
};


template <>
struct ShapeParser<ARGMAX> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        teq::RankT return_dim;
        eigen::Packer<teq::RankT>().unpack(return_dim, attrs);
        if (return_dim >= teq::rank_cap)
        {
            return teq::Shape();
        }
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        slist[return_dim] = 1;
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<PERMUTE> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        teq::RanksT order;
        eigen::Packer<teq::RanksT>().unpack(order, attrs);
        bool redundant = order.empty() ? true : (order[0] == 0);
        for (size_t i = 1, n = std::min(order.size(), (size_t) teq::rank_cap);
            i < n && redundant; ++i)
        {
            redundant = redundant && (order[i] == (order[i-1] + 1));
        }
        if (redundant)
        {
            global::debug("permuting with same "
                "dimensions ... treating as identity");
        }
        return redundant;

    }
};


template <>
struct ShapeParser<PERMUTE> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        teq::RanksT order;
        eigen::Packer<teq::RanksT>().unpack(order, attrs);
        bool visited[teq::rank_cap];
        std::fill(visited, visited + teq::rank_cap, false);
        for (teq::RankT i = 0, n = std::min(order.size(),
            (size_t) teq::rank_cap); i < n; ++i)
        {
            if (visited[order[i]])
            {
                global::throw_errf("permute does not support repeated orders "
                    "(order=%s)", fmts::to_string(order.begin(), order.end()).c_str());
            }
            visited[order[i]] = true;
        }
        for (teq::RankT i = 0; i < teq::rank_cap; ++i)
        {
            if (false == visited[i])
            {
                order.push_back(i);
            }
        }
        teq::Shape shape = shapes.front();
        teq::DimsT slist(teq::rank_cap, 1);
        for (teq::RankT i = 0; i < teq::rank_cap; ++i)
        {
            slist[i] = shape.at(order[i]);
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<EXTEND> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        auto bcast = eigen::unpack_extend(args.front()->shape(), attrs);
        bool uses_dims = nullptr != attrs.get_attr(
            eigen::Packer<teq::DimsT>().get_key());
        bool redundant = false == bool(bcast) || (uses_dims &&
            (bcast->empty() || std::all_of(bcast->begin(), bcast->end(),
            [](teq::DimT d) { return 1 == d; })));
        if (redundant)
        {
            global::debug("extending with nothing... treating as identity");
        }
        return redundant;

    }
};


template <>
struct ShapeParser<EXTEND> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        teq::Shape shape = shapes.front();
        teq::DimsT bcast;
        if (auto bopt = eigen::unpack_extend(shape, attrs))
        {
            bcast = *bopt;
        }
        if (std::any_of(bcast.begin(), bcast.end(),
            [](teq::DimT d) { return 0 == d; }))
        {
            global::throw_errf("cannot extend using zero dimensions %s",
                fmts::to_string(bcast.begin(), bcast.end()).c_str());
        }
        teq::DimsT slist(shape.begin(), shape.end());
        for (size_t i = 0, nbcasts = bcast.size(); i < nbcasts; ++i)
        {
            if (bcast.at(i) > 1 && shape.at(i) > 1)
            {
                global::throw_errf("cannot extend non-singular dimension %d of "
                    "shape %s: bcast=%s", i, shape.to_string().c_str(),
                    fmts::to_string(bcast.begin(), bcast.end()).c_str());
            }
            slist[i] *= bcast[i];
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<RESHAPE> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        teq::Shape outshape;
        eigen::Packer<teq::Shape>().unpack(outshape, attrs);
        bool redundant = outshape.compatible_after(args.front()->shape(), 0);
        if (redundant)
        {
            global::debugf("outshape is the same shape as inshape "
                "%s... treating as identity", outshape.to_string().c_str());
        }
        return redundant;

    }
};


template <>
struct ShapeParser<RESHAPE> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        teq::Shape outshape;
        eigen::Packer<teq::Shape>().unpack(outshape, attrs);
        if (shapes.front().n_elems() != outshape.n_elems())
        {
            global::throw_errf("cannot RESHAPE with shapes of different sizes "
                "%d (shape %s) and %d (shape %s)",
                shapes.front().n_elems(),
                shapes.front().to_string().c_str(),
                outshape.n_elems(),
                outshape.to_string().c_str());
        }
        return outshape;

    }
};


template <>
struct FuncOpt<SLICE> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        eigen::PairVecT<teq::DimT> extents;
        eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, attrs);
        if (std::any_of(extents.begin(), extents.end(),
            [](std::pair<teq::DimT,teq::DimT> ext){ return ext.second == 0; }))
        {
            global::fatalf("cannot create slice with 0 dimensions "
                "(second value of extents) (extents=%s)",
                eigen::to_string(extents).c_str());
        }
        teq::Shape shape = args.front()->shape();
        bool redundant = true;
        for (size_t i = 0, n = std::min(extents.size(),
            (size_t) teq::rank_cap); i < n && redundant; ++i)
        {
            auto& exts = extents[i];
            redundant = redundant && exts.first == 0 &&
                exts.second > shape.at(i);
        }
        if (redundant)
        {
            global::debugf("slice parameter covers whole tensor... "
                "treating as identity: (extents=%s)",
                eigen::to_string(extents).c_str());
        }
        return redundant;

    }
};


template <>
struct ShapeParser<SLICE> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        eigen::PairVecT<teq::DimT> extents;
        eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(extents, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (size_t i = 0, n = std::min(extents.size(),
            (size_t) teq::rank_cap); i < n; ++i)
        {
            teq::DimT offsets = extents[i].first;
            if (offsets < shape.at(i))
            {
                slist[i] = std::min(extents[i].second, (teq::DimT) (shape.at(i) - offsets));
            }
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<PAD> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        eigen::PairVecT<teq::DimT> paddings;
        eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, attrs);
        bool redundant = paddings.empty() || std::all_of(paddings.begin(), paddings.end(),
        [](std::pair<teq::DimT,teq::DimT> pad)
        {
            return pad.first == 0 && pad.second == 0;
        });
        if (redundant)
        {
            global::debugf("padding are all zero... "
                "treating as identity: (paddings=%s)",
                eigen::to_string(paddings).c_str());
        }
        return redundant;

    }
};


template <>
struct ShapeParser<PAD> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        eigen::PairVecT<teq::DimT> paddings;
        eigen::Packer<eigen::PairVecT<teq::DimT>>().unpack(paddings, attrs);
        teq::Shape shape = shapes.front();
        teq::DimsT slist(shape.begin(), shape.end());
        for (size_t i = 0, n = std::min(paddings.size(),
            (size_t) teq::rank_cap); i < n; ++i)
        {
            if (slist[i] > 0)
            {
                slist[i] += paddings[i].first + paddings[i].second;
            }
        }
        return teq::Shape(slist);

    }
};


template <>
struct ShapeParser<STRIDE> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        teq::DimsT incrs;
        eigen::Packer<teq::DimsT>().unpack(incrs, attrs);

        teq::Shape shape = shapes.front();
        std::vector<double> coords(teq::rank_cap, 1);
        size_t n = std::min(incrs.size(), (size_t) teq::rank_cap);
        for (size_t i = 0; i < n; ++i)
        {
            coords[i] = incrs[i];
        }
        teq::DimsT slist(shape.begin(), shape.end());
        for (size_t i = 0; i < n; ++i)
        {
            slist[i] = std::round((double) slist[i] / incrs[i]);
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<SCATTER> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        teq::Shape outshape;
        eigen::Packer<teq::Shape>().unpack(outshape, attrs);
        bool redundant = outshape.compatible_after(args.front()->shape(), 0);
        if (redundant)
        {
            global::debugf("outshape is the same shape as inshape "
                "%s... treating as identity", outshape.to_string().c_str());
        }
        return redundant;

    }
};


template <>
struct ShapeParser<SCATTER> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        if (shapes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        teq::Shape outshape;
        eigen::Packer<teq::Shape>().unpack(outshape, attrs);
        return outshape;

    }
};


template <>
struct FuncOpt<ADD> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        bool redundant = args.size() < 2;
        if (redundant)
        {
            // assuming empty args is handled before
            global::debug("redundantly performing nnary op on a single arg... "
                "treating as identity");
        }
        return redundant;

    }
};


template <>
struct FuncOpt<MUL> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        bool redundant = args.size() < 2;
        if (redundant)
        {
            // assuming empty args is handled before
            global::debug("redundantly performing nnary op on a single arg... "
                "treating as identity");
        }
        return redundant;

    }
};


template <>
struct ShapeParser<MATMUL> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        eigen::PairVecT<teq::RankT> ranks;
        eigen::Packer<eigen::PairVecT<teq::RankT>>().unpack(ranks, attrs);

        // check common dimensions
        std::array<bool,teq::rank_cap> acommon;
        std::array<bool,teq::rank_cap> bcommon;
        std::fill(acommon.begin(), acommon.end(), false);
        std::fill(bcommon.begin(), bcommon.end(), false);
        teq::Shape ashape = shapes[0];
        teq::Shape bshape = shapes[1];
        for (const std::pair<teq::RankT,teq::RankT>& coms : ranks)
        {
            if (ashape.at(coms.first) != bshape.at(coms.second))
            {
                global::throw_errf("invalid shapes %s and %s do not match "
                    "common dimensions %s", ashape.to_string().c_str(),
                    bshape.to_string().c_str(),
                    eigen::to_string(ranks).c_str());
            }
            if (acommon[coms.first] || bcommon[coms.second])
            {
                global::throw_errf("contraction dimensions %s must be unique for "
                    "each side", eigen::to_string(ranks).c_str());
            }
            acommon[coms.first] = bcommon[coms.second] = true;
        }
        teq::DimsT alist = teq::narrow_shape(ashape);
        teq::DimsT blist = teq::narrow_shape(bshape);
        teq::DimsT outlist;
        outlist.reserve(2 * ranks.size());
        for (teq::RankT i = 0, n = blist.size(); i < n; ++i)
        {
            if (false == bcommon[i])
            {
                outlist.push_back(blist.at(i));
            }
        }
        for (teq::RankT i = 0, n = alist.size(); i < n; ++i)
        {
            if (false == acommon[i])
            {
                outlist.push_back(alist.at(i));
            }
        }
        return teq::Shape(outlist);

    }
};


template <>
struct ShapeParser<CONV> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        teq::RanksT ranks;
        eigen::Packer<teq::RanksT>().unpack(ranks, attrs);

        size_t n = std::min(ranks.size(), (size_t) teq::rank_cap);
        teq::Shape kernelshape = shapes[1];
        if (std::any_of(kernelshape.begin() + n, kernelshape.end(),
            [](teq::DimT d) { return d > 1; }))
        {
            global::throw_errf("cannot have ambiguous ranks not specified in "
                "kernelshape %s (ranks=%s)", kernelshape.to_string().c_str(),
                fmts::to_string(ranks.begin(), ranks.end()).c_str());
        }
        teq::Shape imgshape = shapes[0];
        teq::DimsT slist(imgshape.begin(), imgshape.end());
        for (size_t i = 0; i < n; ++i)
        {
            teq::DimT& sdim = slist[ranks[i]];
            teq::DimT kdim = kernelshape.at(i);
            // treat as ambiguous if either dimension is ambiguous
            if (0 == sdim || 0 == kdim)
            {
                sdim = 0;
            }
            else
            {
                if (kdim > sdim)
                {
                    global::throw_errf("cannot convolve a kernel of shape %s against "
                        "smaller image of shape %s at dimensions (shape:kernel=%d:%d)",
                        kernelshape.to_string().c_str(),
                        imgshape.to_string().c_str(), ranks[i], i);
                }
                sdim -= kdim - 1;
            }
        }
        return teq::Shape(slist);

    }
};


template <>
struct FuncOpt<CONCAT> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
        bool redundant = args.size() < 2;
        if (redundant)
        {
            // assuming empty args is handled before
            global::debug("redundantly performing nnary op on a single arg... "
                "treating as identity");
        }
        return redundant;

    }
};


template <>
struct ShapeParser<CONCAT> final
{
    teq::Shape operator() (const marsh::Maps& attrs, const teq::ShapesT& shapes)
    {
        //
        teq::RankT axis;
        eigen::Packer<teq::RankT>().unpack(axis, attrs);
        teq::Shape initshape = shapes.front();
        for (auto it = shapes.begin() + 1, et = shapes.end();
            it != et; ++it)
        {
            if (false == initshape.compatible_before(*it, axis) ||
                false == initshape.compatible_after(*it, axis + 1))
            {
                global::throw_errf("cannot group concat incompatible shapes %s and %s "
                    "along axis %d", initshape.to_string().c_str(), it->to_string().c_str(), axis);
            }
        }
        if (shapes.size() > 2)
        {
            if (std::any_of(shapes.begin(), shapes.end(),
                [axis](teq::Shape shape)
                { return shape.at(axis) != 1; }))
            {
                global::throw_err("cannot group concat shapes "
                    "with dimension that is not one");
            }
            teq::DimsT slist(initshape.begin(), initshape.end());
            slist[axis] = shapes.size();
            return teq::Shape(slist);
        }
        teq::Shape backshape = shapes[1];
        teq::DimsT slist(initshape.begin(), initshape.end());
        if (slist[axis] == 0 || backshape.at(axis) == 0)
        {
            slist[axis] = 0;
        }
        else
        {
            slist[axis] += backshape.at(axis);
        }
        return teq::Shape(slist);

    }
};


template <>
struct TypeParser<ASSIGN> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return dtypes.front();

    }
};


template <>
struct TypeParser<ASSIGN_ADD> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return dtypes.front();

    }
};


template <>
struct TypeParser<ASSIGN_SUB> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return dtypes.front();

    }
};


template <>
struct TypeParser<ASSIGN_MUL> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return dtypes.front();

    }
};


template <>
struct TypeParser<ASSIGN_DIV> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        return dtypes.front();

    }
};


template <>
struct FuncOpt<CAST> final
{
    template <typename T>
    bool operator() (const marsh::Maps& attrs, const teq::TensptrsT& args)
    {
        //
      auto argtype = (egen::_GENERATED_DTYPE)
          args.front()->get_meta().type_code();
      bool redundant = argtype == egen::get_type<T>();
      if (redundant)
      {
          global::debugf("redundantly casting to same type %s",
              egen::name_type(argtype).c_str());
      }
      return redundant;

    }
};


template <>
struct TypeParser<CAST> final
{
    egen::_GENERATED_DTYPE operator() (const marsh::Maps& attrs, const eigen::DTypesT& dtypes)
    {
        //
        if (dtypes.empty())
        {
            global::fatal(eigen::no_argument_err);
        }
        if (attrs.get_attr(eigen::Packer<egen::_GENERATED_DTYPE>::key_))
        {
            egen::_GENERATED_DTYPE out;
            eigen::Packer<egen::_GENERATED_DTYPE>().unpack(out, attrs);
            return out;
        }
        return dtypes.front();

    }
};


}

#endif // _GENERATED_OPCODES_HPP
