#include "estd/contain.hpp"
#include "internal/eigen/generated/opcode.hpp"


#ifdef _GENERATED_OPCODES_HPP

namespace egen
{

static const std::unordered_map<_GENERATED_OPCODE,std::string,estd::EnumHash> code2name =
{
    //>>> code2names
    { IDENTITY, "IDENTITY" },
    { ABS, "ABS" },
    { NEG, "NEG" },
    { SIN, "SIN" },
    { COS, "COS" },
    { TAN, "TAN" },
    { EXP, "EXP" },
    { LOG, "LOG" },
    { SQRT, "SQRT" },
    { ROUND, "ROUND" },
    { SIGMOID, "SIGMOID" },
    { TANH, "TANH" },
    { SQUARE, "SQUARE" },
    { CUBE, "CUBE" },
    { RAND_UNIF, "RAND_UNIF" },
    { REVERSE, "REVERSE" },
    { REDUCE_SUM, "REDUCE_SUM" },
    { REDUCE_PROD, "REDUCE_PROD" },
    { REDUCE_MIN, "REDUCE_MIN" },
    { REDUCE_MAX, "REDUCE_MAX" },
    { ARGMAX, "ARGMAX" },
    { PERMUTE, "PERMUTE" },
    { EXTEND, "EXTEND" },
    { RESHAPE, "RESHAPE" },
    { SLICE, "SLICE" },
    { PAD, "PAD" },
    { STRIDE, "STRIDE" },
    { SCATTER, "SCATTER" },
    { POW, "POW" },
    { ADD, "ADD" },
    { SUB, "SUB" },
    { MUL, "MUL" },
    { DIV, "DIV" },
    { MIN, "MIN" },
    { MAX, "MAX" },
    { EQ, "EQ" },
    { NEQ, "NEQ" },
    { LT, "LT" },
    { GT, "GT" },
    { MATMUL, "MATMUL" },
    { CONV, "CONV" },
    { SELECT, "SELECT" },
    { CONCAT, "CONCAT" },
    { ASSIGN, "ASSIGN" },
    { ASSIGN_ADD, "ASSIGN_ADD" },
    { ASSIGN_SUB, "ASSIGN_SUB" },
    { ASSIGN_MUL, "ASSIGN_MUL" },
    { ASSIGN_DIV, "ASSIGN_DIV" },
    { CAST, "CAST" }
};

static const std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{
    //>>> name2codes
    { "IDENTITY", IDENTITY },
    { "ABS", ABS },
    { "NEG", NEG },
    { "SIN", SIN },
    { "COS", COS },
    { "TAN", TAN },
    { "EXP", EXP },
    { "LOG", LOG },
    { "SQRT", SQRT },
    { "ROUND", ROUND },
    { "SIGMOID", SIGMOID },
    { "TANH", TANH },
    { "SQUARE", SQUARE },
    { "CUBE", CUBE },
    { "RAND_UNIF", RAND_UNIF },
    { "REVERSE", REVERSE },
    { "REDUCE_SUM", REDUCE_SUM },
    { "REDUCE_PROD", REDUCE_PROD },
    { "REDUCE_MIN", REDUCE_MIN },
    { "REDUCE_MAX", REDUCE_MAX },
    { "ARGMAX", ARGMAX },
    { "PERMUTE", PERMUTE },
    { "EXTEND", EXTEND },
    { "RESHAPE", RESHAPE },
    { "SLICE", SLICE },
    { "PAD", PAD },
    { "STRIDE", STRIDE },
    { "SCATTER", SCATTER },
    { "POW", POW },
    { "ADD", ADD },
    { "SUB", SUB },
    { "MUL", MUL },
    { "DIV", DIV },
    { "MIN", MIN },
    { "MAX", MAX },
    { "EQ", EQ },
    { "NEQ", NEQ },
    { "LT", LT },
    { "GT", GT },
    { "MATMUL", MATMUL },
    { "CONV", CONV },
    { "SELECT", SELECT },
    { "CONCAT", CONCAT },
    { "ASSIGN", ASSIGN },
    { "ASSIGN_ADD", ASSIGN_ADD },
    { "ASSIGN_SUB", ASSIGN_SUB },
    { "ASSIGN_MUL", ASSIGN_MUL },
    { "ASSIGN_DIV", ASSIGN_DIV },
    { "CAST", CAST }
};

static const std::unordered_set<_GENERATED_OPCODE,estd::EnumHash> commutatives =
{
    //>>> commcodes
    ADD,
    MUL,
    MIN,
    MAX,
    EQ,
    NEQ
};

static const std::unordered_set<_GENERATED_OPCODE,estd::EnumHash> idempotents =
{
    //>>> idemcodes
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
    ASSIGN
};

std::string name_op (_GENERATED_OPCODE code)
{
    return estd::try_get(code2name, code, "BAD_OP");
}

_GENERATED_OPCODE get_op (const std::string& name)
{
    return estd::try_get(name2code, name, BAD_OP);
}

bool is_commutative (_GENERATED_OPCODE code)
{
    return estd::has(commutatives, code);
}

bool is_commutative (const std::string& name)
{
    if (estd::has(name2code, name))
    {
        return is_commutative(name2code.at(name));
    }
    return false;
}

bool is_idempotent (_GENERATED_OPCODE code)
{
    return estd::has(idempotents, code);
}

bool is_idempotent (const std::string& name)
{
    if (estd::has(name2code, name))
    {
        return is_idempotent(name2code.at(name));
    }
    return false;
}

}

#endif
