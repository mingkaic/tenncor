
#ifndef GLOBAL_MOCK_GENERATOR_HPP
#define GLOBAL_MOCK_GENERATOR_HPP

#include "internal/global/global.hpp"

#include "gmock/gmock.h"

struct MockGenerator final : public global::iGenerator
{
	MOCK_CONST_METHOD0(get_str, std::string(void));

	MOCK_CONST_METHOD2(unif_int, int64_t(const int64_t&,const int64_t&));

	MOCK_CONST_METHOD2(unif_dec, double(const double&,const double&));

	MOCK_CONST_METHOD2(norm_dec, double(const double&,const double&));

	MOCK_CONST_METHOD0(get_strgen, global::GenF<std::string>(void));

	MOCK_CONST_METHOD2(unif_intgen, global::GenF<int64_t>(const int64_t&,const int64_t&));

	MOCK_CONST_METHOD2(unif_decgen, global::GenF<double>(const double&,const double&));

	MOCK_CONST_METHOD2(norm_decgen, global::GenF<double>(const double&,const double&));
};

#endif // GLOBAL_MOCK_GENERATOR_HPP
