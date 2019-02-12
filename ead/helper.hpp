#include "ead/funcarg.hpp"
#include "ead/functor.hpp"

#ifndef EAD_HELPER_HPP
#define EAD_HELPER_HPP

namespace ead
{

template <typename T>
NodeptrT<T> reduce_help (ade::Opcode opcode, NodeptrT<T> tens,
	uint8_t start, uint8_t end)
{
	std::vector<ade::DimT> coords(end - start);
	std::iota(coords.begin(), coords.end(), start);
	return make_functor<T>(opcode, {
		reduce_map(tens, start, coords)
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
		FuncArg<T>(a, left_shaper, false, nullptr),
		FuncArg<T>(b, right_shaper, false, nullptr)
	});
}

}

#endif // EAD_HELPER_HPP
