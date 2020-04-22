#include "teq/itensor.hpp"
#include "teq/isession.hpp"

#ifndef TEQ_MOCK_DEVICEREF_HPP
#define TEQ_MOCK_DEVICEREF_HPP

struct MockDeviceRef : public teq::iDeviceRef
{
	MockDeviceRef (void) = default;

	MockDeviceRef (const std::vector<double>& data) : data_(data) {}

	void* data (void) override
	{
		return data_.data();
	}

	const void* data (void) const override
	{
		return data_.data();
	}

	std::vector<double> data_;

	bool updated_ = false;
};

struct MockDevice : public teq::iDevice
{
	void calc (teq::iTensor& tens) override
	{
		static_cast<MockDeviceRef&>(tens.device()).updated_ = true;
	}
};

#endif // TEQ_MOCK_DEVICEREF_HPP
