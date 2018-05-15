//
//  proto_init.cpp
//  lead
//

#include "lead/proto_init.hpp"

#ifdef LEAD_PB_BUILDER_HPP

namespace lead
{

PbBuilder::PbBuilder (const TensorPb& pb) :
	pb_(pb) {}

clay::Tensor* PbBuilder::get (void) const
{
	auto shapepb = pb_.shape();
	clay::Shape shape(std::vector<size_t>{shapepb.begin(), shapepb.end()});
	TensorT dtype = pb_.type();
	std::shared_ptr<char> data = unpack_data(pb_.data(), dtype);
	return std::make_unique<clay::Tensor>(data, shape, (clay::DTYPE) dtype);
}

clay::Tensor* PbBuilder::get (clay::Shape shape) const
{
	std::unique_ptr<clay::Tensor> out = nullptr;
	auto shapepb = pb_.shape();
	clay::Shape inshape(std::vector<size_t>{shapepb.begin(), shapepb.end()});
	if (inshape.is_compatible_with(shape))
	{
		TensorT dtype = pb_.type();
		std::shared_ptr<char> data = unpack_data(pb_.data(), dtype);
		out = std::make_unique<clay::Tensor>(data, shape, (clay::DTYPE) dtype);
	}
	return out;
}

}

#endif
