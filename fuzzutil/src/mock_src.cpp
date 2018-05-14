#include "clay/memory.hpp"

#include "fuzzutil/mock_src.hpp"
#include "fuzzutil/sgen.hpp"

#ifdef TESTUTIL_MOCK_SRC_HPP

namespace testutil
{

mock_source::mock_source (testify::fuzz_test* fuzzer) :
	mock_source(random_def_shape(fuzzer),
	(clay::DTYPE) fuzzer->get_int(1, "dtype", 
	{1, clay::DTYPE::_SENTINEL - 1})[0], fuzzer) {}

mock_source::mock_source (clay::Shape shape, clay::DTYPE dtype, testify::fuzz_test* fuzzer)
{
	size_t nbytes = shape.n_elems() * clay::type_size(dtype);
	uuid_ = fuzzer->get_string(nbytes, "mock_src_uuid");

	ptr_ = clay::make_char(nbytes);
	std::memcpy(ptr_.get(), uuid_.c_str(), nbytes);

	state_ = {ptr_, shape, dtype};
}

mock_source::mock_source (std::shared_ptr<char> ptr, clay::Shape shape, clay::DTYPE dtype) :
	state_(ptr, shape, dtype), ptr_(ptr)
{
	if (nullptr != ptr && shape.is_fully_defined() && clay::DTYPE::BAD != dtype)
	{
		size_t nbytes = shape.n_elems() * clay::type_size(dtype);
		uuid_ = std::string(ptr.get(), nbytes);
	}
}

bool mock_source::read_data (clay::State& dest) const
{
	bool success = false == uuid_.empty() &&
		dest.dtype_ == state_.dtype_ &&
		dest.shape_.is_compatible_with(state_.shape_);
	if (success)
	{
		std::memcpy((void*) dest.data_.lock().get(), ptr_.get(), uuid_.size());
	}
	return success;
}

}

#endif
