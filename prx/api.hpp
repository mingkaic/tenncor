#include "ead/generated/api.hpp"

#include "prx/codes.hpp"
#include "prx/subgraph.hpp"

namespace prx
{

template <typename T>
ade::TensptrT fully_connected (ade::TensT inputs,
    ade::TensT weights, ade::TensptrT bias)
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
    auto out = age::matmul(ead::to_node<T>(inputs[0]),
        ead::to_node<T>(weights[0]));
    for (size_t i = 1; i < n; ++i)
    {
        out = age::add(age::matmul(inputs[i],
            ead::to_node<T>(weights_[i])));
    }
	const ade::Shape& shape = out->shape();
    // expecting bias to be of shape <x,1,1,...,1>
    inputs.insert(inputs.end(), weights.begin(), weights.end());
    inputs.push_back(bias)
    out = age::add(out, age::extend(
        ead::to_node<T>(bias), 1, {shape.at(1)}));
    return Subgraph::get(ade::Opcode{"FULL_CONN", FULL_CONN},
        out->get_tensor(), inputs);
}

template <typename T>
ade::TensptrT conv_layer (ade::TensptrT inputs,
    ade::TensptrT kernel, ade::TensptrT bias)
{
    auto out = age::convolution(ead::to_node<T>(inputs[0]),
        ead::to_node<T>(kernel));
	// const ade::Shape& shape = out->shape();
    // // expecting bias to be of shape <x,1,1,...,1>
    // out = age::add(out, age::extend(
    //     ead::to_node<T>(bias), 1, {shape.at(1)}));
    // return Subgraph::get(ade::Opcode{"FULL_CONN", FULL_CONN},
    //     out->get_tensor(), weights);
    return out; // todo: account for filters and batches
}

template <typename T>
ade::TensptrT softmax (ade::TensptrT input)
{
	auto num = age::exp(ead::to_node<T>(input));
	auto denom = age::reduce_sum(num);
	ade::Shape shape = input->shape();
	auto out = age::div(num, age::extend(denom, 0,
		std::vector<uint8_t>(shape.begin(), shape.end())));
    return Subgraph::get(ade::Opcode{"SOFT_MAX", SOFT_MAX},
        out->get_tensor(), {input});
}

}
