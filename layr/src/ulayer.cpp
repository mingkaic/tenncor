#include "layr/ulayer.hpp"

#ifdef LAYR_ULAYER_HPP

namespace layr
{

LayerptrT ULayerBuilder::build (void) const
{
	return std::make_shared<ULayer>(utype_, params_, label_);
}

LinkptrT softmax_from_param (LinkptrT input, LinkptrT params)
{
	if (false == params->shape().compatible_after(teq::Shape{}, 0))
	{
		logs::fatalf("Unknown softmax layer parameter %s of shape %s",
			params->to_string().c_str(),
			params->shape().to_string().c_str());
	}
	teq::RankT dim = *((PybindT*) params->data());
	return tenncor::softmax<PybindT>(input, dim, 1);
}

LinkptrT maxpool_from_param (LinkptrT input, LinkptrT params)
{
	if (false == params->shape().compatible_after(teq::Shape({2}), 0))
	{
		logs::fatalf("Unknown maxpool layer parameter %s of shape %s",
			params->to_string().c_str(),
			params->shape().to_string().c_str());
	}
	auto raw_params = (PybindT*) params->data();
	return tenncor::nn::max_pool2d<PybindT>(input,
		{(teq::RankT) raw_params[0], (teq::RankT) raw_params[1]});
}

LinkptrT meanpool_from_param (LinkptrT input, LinkptrT params)
{
	if (false == params->shape().compatible_after(teq::Shape({2}), 0))
	{
		logs::fatalf("Unknown meanpool layer parameter %s of shape %s",
			params->to_string().c_str(),
			params->shape().to_string().c_str());
	}
	auto raw_params = (PybindT*) params->data();
	return tenncor::nn::mean_pool2d<PybindT>(input,
		{(teq::RankT) raw_params[0], (teq::RankT) raw_params[1]});
}

UnaryptrT sigmoid (std::string label)
{
	return std::make_shared<ULayer>(sigmoid_layer_key, nullptr, label);
}

UnaryptrT tanh (std::string label)
{
	return std::make_shared<ULayer>(tanh_layer_key, nullptr, label);
}

UnaryptrT relu (std::string label)
{
	return std::make_shared<ULayer>(relu_layer_key, nullptr, label);
}

UnaryptrT softmax (teq::RankT dim, std::string label)
{
	return std::make_shared<ULayer>(softmax_layer_key,
		eteq::make_constant_scalar<PybindT>(dim, {}), label);
}

UnaryptrT maxpool2d (
	std::pair<teq::DimT,teq::DimT> dims, std::string label)
{
	return std::make_shared<ULayer>(maxpool2d_layer_key,
		eteq::make_constant<PybindT>(
			std::array<PybindT,2>{
				(PybindT) dims.first,
				(PybindT) dims.second}.data(),
			teq::Shape({2})), label);
}

UnaryptrT meanpool2d (
	std::pair<teq::DimT,teq::DimT> dims, std::string label)
{
	return std::make_shared<ULayer>(meanpool2d_layer_key,
		eteq::make_constant<PybindT>(
			std::array<PybindT,2>{
				(PybindT) dims.first,
				(PybindT) dims.second}.data(),
			teq::Shape({2})), label);
}

}

#endif
