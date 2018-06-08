//
//  pb_build.cpp
//  lead
//

#include "lead/include/pb_build.hpp"

#ifdef LEAD_PB_BUILDER_HPP

namespace lead
{

PbBuilder::PbBuilder (const tenncor::TensorPb& pb) :
	pb_(pb) {}

clay::TensorPtrT PbBuilder::get (void) const
{
	auto shapepb = pb_.shape();
	clay::Shape shape(std::vector<size_t>{shapepb.begin(), shapepb.end()});
	tenncor::TensorT dtype = pb_.type();
	std::shared_ptr<char> data = unpack_data(pb_.data(), dtype);
	return std::make_unique<clay::Tensor>(data, shape, (clay::DTYPE) dtype);
}

clay::TensorPtrT PbBuilder::get (clay::Shape shape) const
{
	clay::TensorPtrT out = nullptr;
	auto shapepb = pb_.shape();
	clay::Shape inshape(std::vector<size_t>{shapepb.begin(), shapepb.end()});
	if (inshape.is_compatible_with(shape))
	{
		tenncor::TensorT dtype = pb_.type();
		std::shared_ptr<char> data = unpack_data(pb_.data(), dtype);
		out = std::make_unique<clay::Tensor>(data, shape, (clay::DTYPE) dtype);
	}
	return out;
}

}

#endif
