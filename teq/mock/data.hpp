#include "teq/idata.hpp"

#ifndef TEQ_MOCK_DATA_HPP
#define TEQ_MOCK_DATA_HPP

struct MockData : public teq::iData
{
    MockData (teq::Shape shape, std::vector<double> data) :
        data_(data), shape_(shape) {}

    virtual ~MockData (void) = default;

    void* data (void) override
    {
        return data_.data();
    }

    const void* data (void) const override
    {
        return data_.data();
    }

    teq::Shape data_shape (void) const override
    {
        return shape_;
    }

    size_t type_code (void) const override
    {
        return 0;
    }

    std::string type_label (void) const override
    {
        return "double";
    }

    size_t nbytes (void) const override
    {
        return sizeof(double) * data_.size();
    }

    std::vector<double> data_;

    teq::Shape shape_;
};

#endif // TEQ_MOCK_DATA_HPP
