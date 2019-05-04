#include "ade/funcarg.hpp"

#ifdef ADE_FUNCARG_HPP

namespace ade
{

Shape apply_shaper (const CoordptrT& shaper, Shape inshape)
{
	if (nullptr != shaper)
	{
		CoordT out;
		CoordT in;
		std::copy(inshape.begin(), inshape.end(), in.begin());
		shaper->forward(out.begin(), in.begin());
		std::vector<DimT> slist(rank_cap);
		std::transform(out.begin(), out.end(), slist.begin(),
			[](CDimT cd) -> DimT
			{
				if (cd < 0)
				{
					cd = -cd - 1;
				}
				return std::round(cd);
			});
		inshape = Shape(slist);
	}
	return inshape;
}

FuncArg identity_map (TensptrT tensor)
{
	return FuncArg(tensor, identity);
}

FuncArg reduce_1d_map (TensptrT tensor, uint8_t rank)
{
	Shape shape = tensor->shape();
	std::vector<DimT> indices(rank_cap);
	auto bt = indices.begin();
	auto it = bt + rank;
	std::iota(bt, it, 0);
	std::iota(it, indices.end(), rank + 1);
	indices[rank_cap - 1] = rank;
	return FuncArg(tensor, CoordptrT(
		reduce(rank, {shape.at(rank)})->
		connect(*permute(indices))));
}

FuncArg reduce_map (TensptrT tensor, uint8_t rank, std::vector<DimT> red)
{
	return FuncArg(tensor, reduce(rank, red));
}

FuncArg extend_map (TensptrT tensor, uint8_t rank, std::vector<DimT> ext)
{
	return FuncArg(tensor, extend(rank, ext));
}

FuncArg permute_map (TensptrT tensor, std::vector<uint8_t> order)
{
	return FuncArg(tensor, permute(order));
}

FuncArg flip_map (TensptrT tensor, uint8_t dim)
{
	return FuncArg(tensor, flip(dim));
}

ArgsT to_args (TensT tens)
{
	ArgsT args;
	std::transform(tens.begin(), tens.end(), std::back_inserter(args),
		[](TensptrT& ten)
		{
			return identity_map(ten);
		});
	return args;
}

}

#endif
