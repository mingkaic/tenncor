
#ifndef ETEQ_CASTER_HPP
#define ETEQ_CASTER_HPP

#include "internal/eigen/eigen.hpp"

namespace eteq
{

template <egen::_GENERATED_OPCODE OPCODE>
struct TypeCaster final
{
	template <typename T>
	teq::TensptrsT cast (const teq::TensptrsT& children) const
	{
		marsh::Maps attrs;
		auto type = egen::get_type<T>();
		eigen::pack_attr(attrs, type);
		teq::TensptrsT outs;
		outs.reserve(children.size());
		std::transform(children.begin(), children.end(),
		std::back_inserter(outs),
		[&](teq::TensptrT child)
		{
			if (child->get_meta().type_code() != type)
			{
				return teq::TensptrT(Functor<T>::get(
					egen::CAST, {child}, std::move(attrs)));
			}
			return child;
		});
		return outs;
	}
};

template <>
struct TypeCaster<egen::CAST> final
{
	template <typename T>
	teq::TensptrsT cast (const teq::TensptrsT& children) const
	{
		return children;
	}
};

}

#endif // ETEQ_CASTER_HPP
