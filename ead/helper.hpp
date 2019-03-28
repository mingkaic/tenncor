#include "ead/functor.hpp"

#ifndef EAD_HELPER_HPP
#define EAD_HELPER_HPP

namespace ead
{

template <typename T>
NodeptrT<T> reduce_help (ade::Opcode opcode,
	NodeptrT<T> tens, uint8_t start, uint8_t end)
{
	std::vector<ade::DimT> coords;
	ade::Shape inshape = tens->shape();
	for (size_t i = start; i < end; ++i)
	{
		if (inshape.at(i) > 1)
		{
			coords.push_back(i);
		}
	}
	return make_functor<T>(opcode, {
		reduce_map(tens, start, coords)
	});
}

template <typename T>
NodeptrT<T> reduce_1d_helper (ade::Opcode opcode,
	NodeptrT<T> tens, uint8_t dim)
{
	std::vector<ade::DimT> indices(ade::rank_cap);
	auto bt = indices.begin();
	auto it = bt + dim;
	std::iota(bt, it, 0);
	std::iota(it, indices.end(), dim + 1);
	indices[ade::rank_cap - 1] = dim;

	ade::Shape shape = tens->get_tensor()->shape();
	std::vector<ade::DimT> slist = {shape.at(dim)};
	ade::CoordptrT rshaper(ade::reduce(dim, slist)->connect(
		*ade::permute(indices)));

	return make_functor<T>(opcode, {
		FuncArg<T>(tens, rshaper, reduce({dim}))
	});
}

template <typename T>
NodeptrT<T> get_matmul (NodeptrT<T> a, NodeptrT<T> b)
{
	ade::DimT ncommon = a->get_tensor()->shape().at(0);
	ade::DimT nrow = a->get_tensor()->shape().at(1);
	ade::DimT ncol = b->get_tensor()->shape().at(0);

	ade::CoordptrT left_shaper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			for (uint8_t i = 3; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[2][0] = ncol;
			fwd[1][1] = 1;
			fwd[0][2] = 1.0 / ncommon;
		}
	));

	ade::CoordptrT right_shaper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			for (uint8_t i = 3; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[0][0] = 1;
			fwd[2][1] = nrow;
			fwd[1][2] = 1.0 / ncommon;
		}
	));
	return make_functor<T>(ade::Opcode{"MATMUL", age::MATMUL}, {
		FuncArg<T>(a, left_shaper, nullptr),
		FuncArg<T>(b, right_shaper, nullptr)
	});
}

}

#endif // EAD_HELPER_HPP
