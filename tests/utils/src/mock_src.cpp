#include "mock_src.hpp"

#ifdef TTEST_MOCK_SRC_HPP

namespace testutils
{

mock_data_src::mock_data_src (testify::fuzz_test* fuzzer) :
    type_((TENS_TYPE) fuzzer->get_int(1, "type", {1, N_TYPE - 1})[0]),
    uuid_(fuzzer->get_string(16, "mock_src_uuid")) {}

void mock_data_src::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, nnet::tensorshape shape) const
{
    outptr = nnutils::make_svoid(uuid_.size());
    std::memcpy(outptr.get(), &uuid_[0], uuid_.size());
    type = type_;

    label_incr("get_data");
    std::stringstream ss;
    print_shape(shape, ss);
    set_label("get_data", ss.str());
}

nnet::idata_src* mock_data_src::clone_impl (void) const
{
    return nullptr;
}

}

#endif
