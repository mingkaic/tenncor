#include <string>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "ade/itensor.hpp"

#ifndef OPT_MATCH_CAND_HPP
#define OPT_MATCH_CAND_HPP

namespace opt
{

namespace match
{

using AnyMapT = std::unordered_map<std::string,ade::iTensor*>;

enum CAND_TYPE
{
    SCALAR = 0,
    INTERM,
    CONVRT,
};

struct Symbol final
{
    CAND_TYPE type_;

    // SCALAR -> scalar label
    // INTERM -> intermediate id
    // CONVRT -> conversion ref
    std::string reference_;
};

struct SymbolHash final
{
    size_t operator() (const Symbol& sym) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, sym.type_);
        boost::hash_combine(seed, sym.reference_);
        return seed;
    }
};

inline bool operator == (const Symbol& lhs, const Symbol& rhs)
{
    return lhs.type_ == rhs.type_ && lhs.reference_ == rhs.reference_;
}

using CandsT = std::unordered_map<Symbol,std::vector<AnyMapT>,SymbolHash>;

struct CandArg
{
    ade::iTensor* tensor_;

    CandsT candidates_;

    ade::CoordptrT shaper_;

    ade::CoordptrT coorder_;
};

using CandArgsT = std::vector<CandArg>;

}

}

#endif // OPT_MATCH_CAND_HPP
