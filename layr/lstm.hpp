// source: https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

std::pair<NodeptrT,NodeptrT> lstm (
	NodeptrT prev_ct, NodeptrT prev_ht, NodeptrT input)
{
	auto combine = prev_ht + input; // extend input to fit
	auto forgot = tenncor::sigmoid(comb);
	auto candidate = tenncor::tanh(comb)
	auto ct = prev_ct * forgot + candidate * forgot;
	auto ht = forgot * tenncor::tanh(ct);

	return {ct, ht};
}

// input = placeholder
// ct = zeros
// ht = zeros

// train(x)
// 	input = x
// 	next_ct, next_ht = lstm(ct, ht, input);
// 	ct.assign(next_ct)
// 	ht.assign(next_ht)

