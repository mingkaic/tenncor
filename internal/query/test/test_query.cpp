
#ifndef DISABLE_QUERY_TEST


#include <google/protobuf/util/json_util.h>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/query/querier.hpp"
#include "internal/query/query.pb.h"


static void parse_attr (query::Attribute& attr, const std::string& jstr)
{
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(jstr, &attr, options));
}


TEST(ATTR, Unknown)
{
	query::Query matcher;
	query::Attribute attr;
	marsh::Number<double> numba_wun(1.11);

	query::QResultsT attr_res;
	EXPECT_FATAL(query::equals(
		attr_res, &numba_wun, attr, matcher),
		"cannot compare unknown attribute");
}


TEST(ATTR, IntEquals)
{
	query::Query matcher;
	query::Attribute attr;
	parse_attr(attr, "{\"inum\":5}");

	marsh::Number<double> numba_wun(1.11);
	marsh::Number<size_t> numba_deux(2);
	marsh::Number<float> numba_tres(3.3);
	marsh::Number<int> numba_cinq(5);

	query::QResultsT attr_res;
	EXPECT_FALSE(query::equals(attr_res, &numba_wun, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &numba_deux, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &numba_tres, attr, matcher));
	EXPECT_TRUE(query::equals(attr_res, &numba_cinq, attr, matcher));
}


TEST(ATTR, DecEquals)
{
	query::Query matcher;
	query::Attribute attr;
	parse_attr(attr, "{\"dnum\":3.3}");

	marsh::Number<double> numba_wun(1.11);
	marsh::Number<size_t> numba_deux(2);
	marsh::Number<float> numba_tres(3.3);
	marsh::Number<int> numba_cinq(5);

	query::QResultsT attr_res;
	EXPECT_FALSE(query::equals(attr_res, &numba_wun, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &numba_deux, attr, matcher));
	EXPECT_TRUE(query::equals(attr_res, &numba_tres, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &numba_cinq, attr, matcher));
}


TEST(ATTR, IntArrEquals)
{
	query::Query matcher;
	query::Attribute attr;
	parse_attr(attr, "{\"iarr\":{\"values\":[5,6]}}");
	query::Attribute attr2;
	parse_attr(attr2, "{\"iarr\":{\"values\":[]}}");

	marsh::NumArray<double> root({5., 6});
	marsh::NumArray<size_t> iroot({5, 6});
	marsh::NumArray<size_t> empty;

	query::QResultsT attr_res;
	EXPECT_FALSE(query::equals(attr_res, &root, attr, matcher));
	EXPECT_TRUE(query::equals(attr_res, &iroot, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &empty, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &root, attr2, matcher));
	EXPECT_FALSE(query::equals(attr_res, &iroot, attr2, matcher));
	EXPECT_TRUE(query::equals(attr_res, &empty, attr2, matcher));
}


TEST(ATTR, DecArrEquals)
{
	query::Query matcher;
	query::Attribute attr;
	parse_attr(attr, "{\"darr\":{\"values\":[5,6]}}");
	query::Attribute attr2;
	parse_attr(attr2, "{\"darr\":{\"values\":[]}}");

	marsh::NumArray<size_t> root({5, 6});
	marsh::NumArray<double> iroot({5, 6});
	marsh::NumArray<double> empty;

	query::QResultsT attr_res;
	EXPECT_FALSE(query::equals(attr_res, &root, attr, matcher));
	EXPECT_TRUE(query::equals(attr_res, &iroot, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &empty, attr, matcher));
	EXPECT_FALSE(query::equals(attr_res, &root, attr2, matcher));
	EXPECT_FALSE(query::equals(attr_res, &iroot, attr2, matcher));
	EXPECT_TRUE(query::equals(attr_res, &empty, attr2, matcher));
}


TEST(ATTR, StrEquals)
{
	query::Query matcher;
	query::Attribute attr;
	parse_attr(attr, "{\"str\":\"numba wun\"}");

	marsh::String s1("smitty werbenjagermanjensen");
	marsh::String s2("numba wun");

	query::QResultsT attr_res;
	EXPECT_FALSE(query::equals(attr_res, &s1, attr, matcher));
	EXPECT_TRUE(query::equals(attr_res, &s2, attr, matcher));
}


// TEST(ATTR, NodeEquals)
// {
// 	//
// }


// TEST(ATTR, LayerEquals)
// {
// 	//
// }


#endif // DISABLE_QUERY_TEST
