
#ifndef TEQ_MOCK_METADATA_HPP
#define TEQ_MOCK_METADATA_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

struct MockMeta final : public teq::iMetadata
{
	MOCK_CONST_METHOD0(type_code, size_t(void));

	MOCK_CONST_METHOD0(type_label, std::string(void));

	MOCK_CONST_METHOD0(type_size, size_t(void));

	MOCK_CONST_METHOD0(state_version, size_t(void));
};

#endif // TEQ_MOCK_METADATA_HPP
