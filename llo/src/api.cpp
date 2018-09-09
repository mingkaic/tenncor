#include "ade/functor.hpp"

#include "llo/api.hpp"

ade::Tensorptr abs (ade::Tensorptr& arg)
{
	return ade::Functor<ade::ABS>::get({arg});
}

ade::Tensorptr neg (ade::Tensorptr& arg)
{
	return ade::Functor<ade::NEG>::get({arg});
}

ade::Tensorptr logic_not (ade::Tensorptr& arg)
{
	return ade::Functor<ade::NOT>::get({arg});
}

ade::Tensorptr sin (ade::Tensorptr& arg)
{
	return ade::Functor<ade::SIN>::get({arg});
}

ade::Tensorptr cos (ade::Tensorptr& arg)
{
	return ade::Functor<ade::COS>::get({arg});
}

ade::Tensorptr tan (ade::Tensorptr& arg)
{
	return ade::Functor<ade::TAN>::get({arg});
}

ade::Tensorptr exp (ade::Tensorptr& arg)
{
	return ade::Functor<ade::EXP>::get({arg});
}

ade::Tensorptr log (ade::Tensorptr& arg)
{
	return ade::Functor<ade::LOG>::get({arg});
}

ade::Tensorptr sqrt (ade::Tensorptr& arg)
{
	return ade::Functor<ade::SQRT>::get({arg});
}

ade::Tensorptr round (ade::Tensorptr& arg)
{
	return ade::Functor<ade::ROUND>::get({arg});
}

ade::Tensorptr flip (ade::Tensorptr& arg, uint8_t dim)
{
	return DirectWrapper<uint8_t>::get(
		ade::Functor<ade::FLIP>::get({arg}), dim);
}

ade::Tensorptr pow (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::POW>::get({a, b});
}

ade::Tensorptr add (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::ADD>::get({a, b});
}

ade::Tensorptr sub (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::SUB>::get({a, b});
}

ade::Tensorptr mul (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::MUL>::get({a, b});
}

ade::Tensorptr div (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::DIV>::get({a, b});
}

ade::Tensorptr eq (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::EQ>::get({a, b});
}

ade::Tensorptr neq (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::NE>::get({a, b});
}

ade::Tensorptr lt (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::LT>::get({a, b});
}

ade::Tensorptr gt (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return ade::Functor<ade::GT>::get({a, b});
}

ade::Tensorptr n_elems (ade::Tensorptr& arg)
{
	return ade::Functor<ade::N_ELEMS>::get({arg});
}

ade::Tensorptr n_dims (ade::Tensorptr& arg, uint8_t dim)
{
	return DirectWrapper<uint8_t>::get(
		ade::Functor<ade::N_DIMS>::get({arg}), dim);
}

ade::Tensorptr argmax (ade::Tensorptr& arg)
{
	return ade::Functor<ade::ARGMAX>::get({arg});
}

ade::Tensorptr rmax (ade::Tensorptr& arg)
{
	return ade::Functor<ade::RMAX>::get({arg});
}

ade::Tensorptr rsum (ade::Tensorptr& arg)
{
	return ade::Functor<ade::RSUM>::get({arg});
}

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b)
{
	return DirectWrapper<uint8_t,uint8_t>::get(
		ade::Functor<ade::MATMUL>::get({a, b}), 1, 1);
}

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	return DirectWrapper<uint8_t,uint8_t>::get(
		ade::Functor<ade::MATMUL,uint8_t,uint8_t>::get({a, b},
			agroup_idx, bgroup_idx), agroup_idx, bgroup_idx);
}

ade::Tensorptr permute (ade::Tensorptr& arg, std::vector<uint8_t> order)
{
	return DirectWrapper<std::vector<uint8_t>>::get(
		ade::Functor<ade::PERMUTE,std::vector<uint8_t>>::get(
			{arg}, order), order);
}

ade::Tensorptr extend (ade::Tensorptr& arg, std::vector<uint8_t> ext)
{
	return ade::Functor<ade::EXTEND,std::vector<ade::DimT>>::get({arg}, ext);
}

ade::Tensorptr reshape (ade::Tensorptr& arg, std::vector<uint8_t> slist)
{
	return ade::Functor<ade::RESHAPE,
		std::vector<ade::DimT>>::get({arg}, slist);
}
