#include "ead/functor.hpp"

#ifndef EAD_HELPER_HPP
#define EAD_HELPER_HPP

namespace ead
{

const std::string commutative_tag = "commutative";

template <typename T>
NodeptrT<T> build_reduce (ade::Opcode opcode,
	NodeptrT<T> tens, uint8_t offset, uint8_t ndims)
{
	// todo: report if offset out of rank_cap
	return make_functor<T>(opcode, {
		reduce_map(tens, offset, ndims)
	});
}

template <typename T>
NodeptrT<T> build_reduce_1d (ade::Opcode opcode,
	NodeptrT<T> tens, uint8_t dim)
{
	// todo: report if offset out of rank_cap
	std::vector<ade::DimT> indices(ade::rank_cap);
	auto bt = indices.begin();
	auto it = bt + dim;
	std::iota(bt, it, 0);
	std::iota(it, indices.end(), dim + 1);
	indices[ade::rank_cap - 1] = dim;

	return make_functor<T>(ade::Opcode{"PERMUTE", age::PERMUTE}, {
		permute_map(make_functor<T>(opcode, {
			reduce_map(tens, dim, 1)
		}), indices)
	});
}

}

#endif // EAD_HELPER_HPP
