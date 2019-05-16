#include "ead/functor.hpp"

#ifndef EAD_HELPER_HPP
#define EAD_HELPER_HPP

namespace ead
{

template <typename T>
NodeptrT<T> build_reduce (ade::Opcode opcode,
	NodeptrT<T> tens, ade::DimT offset, ade::DimT ndims)
{
	// todo: report if offset out of rank_cap
	std::vector<ade::DimT> coords;
	ade::Shape inshape = tens->shape();
	for (size_t i = offset,
		n = std::min<ade::DimT>(offset + ndims, ade::rank_cap); i < n; ++i)
	{
		if (inshape.at(i) > 1)
		{
			coords.push_back(i);
		}
	}
	ade::DimT rank = 0;
	if (coords.size() > 0)
	{
		rank = coords.front();
	}
	return make_functor<T>(opcode, {
		reduce_map(tens, rank, coords)
	});
}

template <typename T>
NodeptrT<T> build_reduce_1d (ade::Opcode opcode,
	NodeptrT<T> tens, ade::DimT dim)
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
			reduce_map(tens, dim, {dim})
		}), indices)
	});
}

template <typename T>
NodeptrT<T> build_matmul (NodeptrT<T> a, NodeptrT<T> b)
{
	ade::Shape ashape = a->get_tensor()->shape();
	ade::Shape bshape = b->get_tensor()->shape();
	ade::DimT ncommon = ashape.at(0);
	ade::DimT nrow = ashape.at(1);
	ade::DimT ncol = bshape.at(0);
	if (ncommon != bshape.at(1))
	{
		logs::fatalf("invalid matmul shapes %s and %s",
			ashape.to_string().c_str(), bshape.to_string().c_str());
	}

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

template <typename T>
NodeptrT<T> build_convolve (NodeptrT<T> input,
	NodeptrT<T> kernel, std::vector<ade::DimT> dims)
{
	ade::Shape inshape = input->get_tensor()->shape();
	ade::Shape kernelshape = kernel->get_tensor()->shape();
	ade::CoordptrT input_shaper(new ade::CoordMap(
		[&kernelshape](ade::MatrixT fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[ade::rank_cap][i] = -kernelshape.at(i) + 1;
			}
		}
	));

	ade::CoordptrT kernel_shaper(new ade::CoordMap(
		[&inshape](ade::MatrixT fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[i][i] = -1;
			}
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd[ade::rank_cap][i] = inshape.at(i) + 1;
			}
		}
	));

	ade::CoordT kernel_dims;
	auto it = kernel_dims.begin();
	std::fill(it, kernel_dims.end(), ade::rank_cap);
	std::copy(dims.begin(), dims.end(), it);
	return make_functor<T>(ade::Opcode{"CONV", age::CONV}, {
		FuncArg<T>(input, input_shaper, nullptr),
		FuncArg<T>(kernel, kernel_shaper, std::make_shared<CoordMap>(
			CONV, kernel_dims, true)),
	});
}

template <typename T>
NodeptrT<T> build_slice (NodeptrT<T> arg,
	ade::DimT offset, ade::DimT extent, ade::DimT dimension)
{
	// todo: report offset out of rank_cap
	ade::CoordT slicings;
	std::fill(slicings.begin(), slicings.end(), ade::rank_cap);
	slicings[0] = offset;
	slicings[1] = extent;
	slicings[2] = dimension;
	return make_functor<T>(ade::Opcode{"SLICE", age::SLICE}, {
		FuncArg<T>(arg,
			std::make_shared<ade::CoordMap>(
				[&](ade::MatrixT fwd)
				{
					for (uint8_t i = 0; i < ade::rank_cap; ++i)
					{
						fwd[i][i] = 1;
					}
					fwd[ade::rank_cap][dimension] =
						extent - arg->shape().at(dimension);
				}),
			std::make_shared<CoordMap>(SLICE, slicings, false)
		)
	});
}

template <typename T>
NodeptrT<T> build_pad (NodeptrT<T> arg,
	std::pair<ade::DimT,ade::DimT> padding, ade::DimT dimension)
{
	// todo: report if dimension out of rank_cap
	ade::CoordT paddings;
	std::fill(paddings.begin(), paddings.end(), ade::rank_cap);
	paddings[0] = padding.first;
	paddings[1] = padding.second;
	paddings[2] = dimension;
	return make_functor<T>(ade::Opcode{"PAD", age::PAD}, {
		FuncArg<T>(arg,
			std::make_shared<ade::CoordMap>(
				[&](ade::MatrixT fwd)
				{
					for (uint8_t i = 0; i < ade::rank_cap; ++i)
					{
						fwd[i][i] = 1;
					}
					fwd[ade::rank_cap][dimension] =
						padding.first + padding.second;
				}),
			std::make_shared<CoordMap>(PAD, paddings, false)
		)
	});
}

#define BUILD_SOFTMAX_DEF(ARG1)\
[&](){\
	auto exarg = exp(ARG1);\
	ade::Shape shape = exarg->shape();\
	auto it = shape.begin() + offset;\
	std::vector<ade::DimT> xlist(it, it + ndims);\
	return div(exarg,extend(reduce_sum(exarg,offset,offset+ndims), offset, xlist));\
}()

// image must be in form [in, width, height, batch]
// kernel must be in form [out, in, width, height]
// see https://www.tensorflow.org/api_docs/python/tf/nn/conv2d specifications
#define BUILD_CONV2D_DEF(ARG1, ARG2)\
[&](){\
	ade::DimT nfilters = ARG2->shape().at(0);\
	ead::NodesT<T> convolveds;\
	convolveds.reserve(nfilters);\
	for (ade::DimT i = 0; i < nfilters; ++i)\
	{\
		auto filter = age::permute(\
			age::slice(ARG2, i, 1, 0),\
			{1, 2, 3, 0});\
		auto conved = age::convolution(ARG1, filter,\
			{0, 1, 2});\
		auto padded = age::pad(conved,\
			{i, nfilters - i - 1}, 0);\
		convolveds.push_back(padded);\
	}\
	auto out = convolveds[0];\
	for (ade::DimT i = 1; i < nfilters; ++i)\
	{\
		out = age::add(out, convolveds[i]);\
	}\
	return out;\
}()

}

#endif // EAD_HELPER_HPP
