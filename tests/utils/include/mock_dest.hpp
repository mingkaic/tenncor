#include "testify/mocker/mocker.hpp"

#include "tensor/data_io.hpp"

#ifndef TTEST_MOCK_DEST_HPP
#define TTEST_MOCK_DEST_HPP

namespace testutils
{

struct mock_data_dest final : public nnet::idata_dest, public testify::mocker
{
	mock_data_dest (void);

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, nnet::tshape shape, size_t idx);

	std::string result_;
	TENS_TYPE type_ = nnet::BAD_T;
	nnet::tshape shape_;
};

}

#endif /* TTEST_MOCK_DEST_HPP */
