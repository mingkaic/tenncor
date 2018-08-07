#include "sand/operator.hpp"

#include "util/error.hpp"

#ifndef SAND_UNARY_HPP
#define SAND_UNARY_HPP

// template <typename T>
// void static_transpose (NodeInfo dest, std::vector<NodeInfo>& srcs,
// 	MetaEncoder::MetaData mdata)
// {
// 	if (1 != srcs.size())
// 	{
// 		handle_error("transpose requires 1 arguments",
// 			ErrArg<size_t>{"num_args", srcs.size()});
// 	}

// 	NodeInfo& src = srcs[0];
// 	NElemT an = src.shape_.n_elems();
// 	NElemT n = dest.shape_.n_elems();

// 	if (an != n)
// 	{
// 		handle_error("transposing src to destination of incompatible size",
// 			ErrArg<size_t>{"ndest", n},
// 			ErrArg<size_t>{"nsrc", an});
// 	}

// 	T* destdata = (T*) dest.data_;
// 	T* srcdata = (T*) src.data_;

// 	NElemT srcx = src.shape_.group(0).n_elems();
// 	NElemT srcy = src.shape_.group(1).n_elems();

// 	// apply transformation
// 	for (NElemT srci = 0; srci < n; ++srci)
// 	{
// 		NElemT row = srci / srcx;
// 		NElemT col = srci % srcx;
// 		NElemT desti = row + col * srcy;
// 		destdata[desti] = srcdata[srci];
// 	}
// }

template <typename T>
void transpose (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	NodeInfo& src = srcs[0];
	NElemT n = src.shape_.n_elems();

	T* destdata = (T*) dest.data_;
	T* srcdata = (T*) src.data_;

	// apply transformation
	std::vector<DimT> coords;
	uint8_t ng0 = mdata[1] - mdata[0];
	uint8_t ng1 = mdata[2] - mdata[1];
	uint8_t ng2 = mdata[3] - mdata[2];
	uint8_t total = mdata[3] - mdata[0];
	DimT buffer[total];
	for (NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(src.shape_, srci);

		std::memcpy(buffer, &coords[mdata[2]], sizeof(DimT) * ng2);
		std::memcpy(buffer + ng2, &coords[mdata[1]], sizeof(DimT) * ng1);
		std::memcpy(buffer + ng2 + ng1, &coords[mdata[0]], sizeof(DimT) * ng0);
		std::memcpy(&coords[mdata[0]], buffer, sizeof(DimT) * total);

		NElemT desti = index(dest.shape_, coords);
		destdata[desti] = srcdata[srci];
	}
}

#endif /* SAND_UNARY_HPP */
