#include "testify/fuzz/fuzz.hpp"
#include "testify/mocker/mocker.hpp"

#include "clay/isource.hpp"

#ifndef TESTUTIL_MOCK_SRC_HPP
#define TESTUTIL_MOCK_SRC_HPP

namespace testutil
{

struct mock_source final : public clay::iSource, public testify::mocker
{
	mock_source (testify::fuzz_test* fuzzer);

	mock_source (clay::Shape shape, clay::DTYPE dtype, testify::fuzz_test* fuzzer);

	mock_source (std::shared_ptr<char> ptr, clay::Shape shape, clay::DTYPE dtype);

	bool read_data (clay::State& dest) const override;

	clay::State state_;

	std::shared_ptr<char> ptr_;

	std::string uuid_;
};

}

#endif /* TESTUTIL_MOCK_SRC_HPP */
