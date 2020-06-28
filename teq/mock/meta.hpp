#include "teq/itensor.hpp"

#ifndef TEQ_MOCK_METADATA_HPP
#define TEQ_MOCK_METADATA_HPP

struct MockMeta : public teq::iMetadata
{
	size_t type_code (void) const override
	{
		return tcode_;
	}

	std::string type_label (void) const override
	{
		return tname_;
	}

	size_t type_size (void) const override
	{
		return 0;
	}

	size_t state_version (void) const override
	{
		return version_;
	}

	size_t tcode_ = 0;

	std::string tname_ = "no_type";

	size_t version_ = 0;
};

#endif // TEQ_MOCK_METADATA_HPP
