#include "fuzz/fuzz.hpp"
#include "mocker/mocker.hpp"

#include "tensor/data_src.hpp"

#ifndef TTEST_MOCK_SRC_HPP
#define TTEST_MOCK_SRC_HPP

namespace testutils
{

struct mock_data_src final : public nnet::idata_src, public testify::mocker
{
	mock_data_src (testify::fuzz_test* fuzzer);

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, nnet::tensorshape shape) const;

	TENS_TYPE type_;

	std::string uuid_;

	virtual nnet::idata_src* clone_impl (void) const;
};

}

#endif /* TTEST_MOCK_SRC_HPP */
