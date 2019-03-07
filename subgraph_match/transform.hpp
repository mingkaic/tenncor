#include <regex>

#include "ade/ifunctor.hpp"

#include "ead/generated/opmap.hpp"

#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef OPT_TRANSFORM_HPP
#define OPT_TRANSFORM_HPP

namespace opt
{

const std::string tuple_delim = "->";

const std::string line_delim = ",,";

using DepthMatrixT = std::vector<std::vector<std::string>>;

struct Transform final
{
    size_t pheight_;

    std::regex pattern_;

    std::string simplification_;
};

using TransformsT = std::vector<Transform>;

DepthMatrixT simplify_depthmatrix (TransformsT& transforms, DepthMatrixT depthm)
{
    for (Transform& transform : transforms)
    {
        std::string serial = depthm[0][0];
        for (size_t i = 1, n = std::min(depthm.size(), transform.pheight_);
            i < n; ++i)
        {
            auto& idepthm = depthm[i];
            serial += line_delim + fmts::join(",", idepthm.begin(), idepthm.end());
        }

        if (std::regex_match(serial, transform.pattern_))
        {
            std::string simplification = std::regex_replace(serial,
                transform.pattern_, transform.simplification_);

            DepthMatrixT simplem;
            auto levels = fmts::split(simplification, line_delim);
            for (std::string level : levels)
            {
                fmts::trim(level);
                if (level.size() > 0)
                {
                    simplem.push_back(fmts::split(level, ","));
                }
            }
            return simplem;
        }
    }
    return depthm;
}

const std::string var_pattern = "variable(%s)";

const std::string cst_pattern = "constant(%s)";

const std::string scalar_fmt = "scalar(%s)";

std::string identify (ade::iTensor* tens)
{
    // pointer to string
    size_t iptr = (size_t) tens;
    return fmts::to_string(iptr);
}

#define __TYPED_ENCODE(realtype)out = typed_encode_leaf<realtype>(\
static_cast<ead::iLeaf<realtype>*>(leaf));

template <typename T>
std::string typed_encode_leaf (ead::iLeaf<T>* leaf)
{
    if (leaf->is_const())
    {
        T* begin = (T*) leaf->data();
        T* end = begin + leaf->shape().n_elems();
        if (std::adjacent_find(begin, end, std::not_equal_to<>()) == end)
        {
            return fmts::sprintf(scalar_fmt,
                fmts::to_string(*begin).c_str());
        }
        return cst_pattern;
    }
    return var_pattern + identify(leaf);
}

std::string encode_leaf (ade::iLeaf* leaf)
{
    std::string out;
    TYPE_LOOKUP(__TYPED_ENCODE, leaf->type_code())
    return out;
}

#undef __TYPED_ENCODE

std::string encode_func (ade::iFunctor* func)
{
    return func->get_opcode().name_ + "(" + identify(func) + ")";
}

}

#endif // OPT_TRANSFORM_HPP
