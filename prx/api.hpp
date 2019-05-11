#include "ead/generated/api.hpp"

#include "prx/codes.hpp"
#include "prx/subgraph.hpp"

#ifndef PRX_API_HPP
#define PRX_API_HPP

namespace prx
{

template <typename T>
ead::NodeptrT<T> fully_connected (ead::NodesT<T> inputs,
	ead::NodesT<T> weights, ead::NodeptrT<T> bias)
{
	if (weights.empty())
	{
		logs::fatal("cannot create a fully connected layer without weights");
	}
	size_t n = inputs.size();
	if (n != weights.size())
	{
		logs::fatalf(
			"number of inputs (%d) must equal the number of weights (%d)",
			n, weights.size());
	}
	auto out = age::matmul(inputs[0], weights[0]);
	for (size_t i = 1; i < n; ++i)
	{
		out = age::add(out, age::matmul(inputs[i], weights[i]));
	}
	const ade::Shape& shape = out->shape();
	// expecting bias to be of shape <x,1,1,...,1>
	inputs.insert(inputs.end(), weights.begin(), weights.end());
	inputs.push_back(bias);
	out = age::add(out, age::extend(bias, 1, {shape.at(1)}));
	return make_subgraph(ade::Opcode{"FULL_CONN", FULL_CONN}, out, inputs);
}

// image must be in form [in, width, height, batch]
// kernel must be in form [out, in, width, height]
// see https://www.tensorflow.org/api_docs/python/tf/nn/conv2d specifications
template <typename T>
ead::NodeptrT<T> conv2d (ead::NodeptrT<T> image, ead::NodeptrT<T> kernel)
{
	ade::DimT nfilters = kernel->shape().at(0);
	ead::NodesT<T> convolveds;
	convolveds.reserve(nfilters);
	for (ade::DimT i = 0; i < nfilters; ++i)
	{
		auto filter = age::permute(
			age::slice(kernel, i, 1, 0),
			{1, 2, 3, 0});
		auto conved = age::convolution(image, filter,
			{0, 1, 2});
		auto padded = age::pad(conved,
			{i, nfilters - i - 1}, 0);
		convolveds.push_back(padded);
	}

	auto out = convolveds[0];
	for (ade::DimT i = 1; i < nfilters; ++i)
	{
		out = age::add(out, convolveds[i]);
	}
	return out; // todo: implement subgraph gradebuilder
	// return make_subgraph(ade::Opcode{"CONV2D", CONV2D},
	// 	out, {image, kernel});
}

template <typename T>
ead::NodeptrT<T> softmax (ead::NodeptrT<T> input)
{
	auto num = age::exp(input);
	auto denom = age::reduce_sum(num);
	ade::Shape shape = input->shape();
	auto out = age::div(num, age::extend(denom, 0,
		std::vector<uint8_t>(shape.begin(), shape.end())));
	return make_subgraph(ade::Opcode{"SOFT_MAX", SOFT_MAX}, out, {input});
}

}

#endif // PRX_API_HPP
