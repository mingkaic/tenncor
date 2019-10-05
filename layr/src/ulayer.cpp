#include "layr/ulayer.hpp"

#ifdef LAYR_ULAYER_HPP

namespace layr
{

LayerptrT ULayerBuilder::build (void) const
{
	return std::make_shared<ULayer>(utype_, label_);
}

NodeptrT softmax_from_layer (const ULayer& layer, NodeptrT input)
{
	return tenncor::softmax<PybindT>(input,
		std::stoi(layer.get_label()), 1);
}

NodeptrT maxpool_from_layer (const ULayer& layer, NodeptrT input)
{
	auto parts = fmts::split(layer.get_label(), ",");
	if (parts.size() != 2)
	{
		logs::fatalf("failed to get maxpool dimensions from %s",
			layer.get_label().c_str());
	}
	return tenncor::nn::max_pool2d<PybindT>(input,
		{std::stoi(parts[0]), std::stoi(parts[1])});
}

NodeptrT meanpool_from_layer (const ULayer& layer, NodeptrT input)
{
	auto parts = fmts::split(layer.get_label(), ",");
	if (parts.size() != 2)
	{
		logs::fatalf("failed to get maxpool dimensions from %s",
			layer.get_label().c_str());
	}
	return tenncor::nn::mean_pool2d<PybindT>(input,
		{std::stoi(parts[0]), std::stoi(parts[1])});
}

UnaryptrT sigmoid (void)
{
	return std::make_shared<ULayer>(sigmoid_layer_key);
}

UnaryptrT tanh (void)
{
	return std::make_shared<ULayer>(tanh_layer_key);
}

UnaryptrT softmax (teq::RankT dim)
{
	return std::make_shared<ULayer>(softmax_layer_key,
		fmts::sprintf("%d", dim));
}

UnaryptrT maxpool2d (std::pair<teq::DimT,teq::DimT> dims)
{
	return std::make_shared<ULayer>(maxpool2d_layer_key,
		fmts::sprintf("%d,%d", dims.first, dims.second));
}

UnaryptrT meanpool2d (std::pair<teq::DimT,teq::DimT> dims)
{
	return std::make_shared<ULayer>(meanpool2d_layer_key,
		fmts::sprintf("%d,%d", dims.first, dims.second));
}

}

#endif
