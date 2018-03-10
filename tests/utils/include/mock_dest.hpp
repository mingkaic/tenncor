#include "mocker/mocker.hpp"

#include "tensor/data_io.hpp"

#ifndef TTEST_MOCK_DEST_HPP
#define TTEST_MOCK_DEST_HPP

namespace testutils
{

struct mock_data_dest final : public nnet::idata_dest, public testify::mocker
{
	mock_data_dest (void) : result_(16, ' ') {}

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, nnet::tensorshape shape, size_t idx)
	{
		label_incr("set_data");
		std::stringstream ss;
		ss << idx;
		set_label("set_data", ss.str());
	
		std::memcpy(&result_[0], data.lock().get(), 16);
		type_ = type;
		shape_ = shape;
	}

	std::string result_;
	TENS_TYPE type_ = BAD_T;
	nnet::tensorshape shape_;
};

}

#endif
