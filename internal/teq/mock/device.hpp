
#ifndef TEQ_MOCK_DEVICEREF_HPP
#define TEQ_MOCK_DEVICEREF_HPP

#include "internal/teq/teq.hpp"

struct MockDeviceRef : public teq::iDeviceRef
{
	MockDeviceRef (void) = default;

	MockDeviceRef (const std::string& data) : data_(data) {}

	template <typename T>
	MockDeviceRef (const std::vector<T>& data)
	{
		auto raw = (char*) data.data();
		data_ = std::string(raw, raw + sizeof(T) * data.size());
	}

	void* data (void) override
	{
		return &data_[0];
	}

	const void* data (void) const override
	{
		return data_.c_str();
	}

	std::string data_;

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
