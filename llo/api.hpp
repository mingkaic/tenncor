#include "ade/functor.hpp"

#include "llo/data.hpp"

extern Session global_sess;

ade::Tensorptr abs (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::ABS>::get({arg});
}

ade::Tensorptr neg (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::NEG>::get({arg});
}

ade::Tensorptr not (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::NOT>::get({arg});
}

ade::Tensorptr sin (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::SIN>::get({arg});
}

ade::Tensorptr cos (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::COS>::get({arg});
}

ade::Tensorptr tan (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::TAN>::get({arg});
}

ade::Tensorptr exp (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::EXP>::get({arg});
}

ade::Tensorptr log (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::LOG>::get({arg});
}

ade::Tensorptr sqrt (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::SQRT>::get({arg});
}

ade::Tensorptr round (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::ROUND>::get({arg});
}

ade::Tensorptr flip (ade::Tensorptr& arg, uint8_t dim, Session& sess = global_sess)
{
    return ade::Functor<ade::FLIP>::get({arg});
}

ade::Tensorptr pow (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::POW>::get({a, b});
}

ade::Tensorptr add (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::ADD>::get({a, b});
}

ade::Tensorptr sub (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::SUB>::get({a, b});
}

ade::Tensorptr mul (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::MUL>::get({a, b});
}

ade::Tensorptr div (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::DIV>::get({a, b});
}

ade::Tensorptr eq (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::EQ>::get({a, b});
}

ade::Tensorptr neq (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::NE>::get({a, b});
}

ade::Tensorptr lt (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::LT>::get({a, b});
}

ade::Tensorptr gt (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::GT>::get({a, b});
}

ade::Tensorptr n_elems (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::N_ELEMS>::get({arg});
}

ade::Tensorptr n_dims (ade::Tensorptr& arg, uint8_t dim, Session& sess = global_sess)
{
    return ade::Functor<ade::N_DIMS>::get({arg});
}

ade::Tensorptr argmax (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::ARGMAX>::get({arg});
}

ade::Tensorptr rmax (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::RMAX>::get({arg});
}

ade::Tensorptr rsum (ade::Tensorptr& arg, Session& sess = global_sess)
{
    return ade::Functor<ade::RSUM>::get({arg});
}

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b, Session& sess = global_sess)
{
    return ade::Functor<ade::MATMUL>::get({a, b});
}

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b, uint8_t agroup_idx, uint8_t bgroup_idx, Session& sess = global_sess)
{
    return ade::Functor<ade::MATMUL,uint8_t,uint8_t>::get({a, b}, agroup_idx, bgroup_idx);
}

ade::Tensorptr permute (ade::Tensorptr& arg, std::vector<uint8_t> order, Session& sess = global_sess)
{
    return ade::Functor<ade::PERMUTE,std::vector<uint8_t>>::get({arg}, order);
}

ade::Tensorptr extend (ade::Tensorptr& arg, std::vector<uint8_t> ext, Session& sess = global_sess)
{
    return ade::Functor<ade::EXTEND,std::vector<ade::DimT>>::get({arg}, ext);
}

ade::Tensorptr reshape (ade::Tensorptr& arg, std::vector<uint8_t> slist, Session& sess = global_sess)
{
    return ade::Functor<ade::RESHAPE,std::vector<ade::DimT>>::get({arg}, slist);
}
