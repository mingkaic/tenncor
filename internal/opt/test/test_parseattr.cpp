
#ifndef DISABLE_OPT_PARSEATTR_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


TEST(PARSEATTR, BadAttr)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	opt::GraphInfo ginfo({});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute badattr;
	std::string fatalmsg = "cannot parse unknown attribute";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(opt::parse_attr(badattr, ginfo), fatalmsg.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(PARSEATTR, Number)
{
	opt::GraphInfo ginfo({});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute iattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"inum\":3}", &iattr, options));

	auto inum = opt::parse_attr(iattr, ginfo);
	auto i = dynamic_cast<marsh::Number<int64_t>*>(inum);
	ASSERT_NE(nullptr, i);
	EXPECT_EQ(3, i->to_int64());

	query::Attribute dattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"dnum\":3.3}", &dattr, options));

	auto dnum = opt::parse_attr(dattr, ginfo);
	auto d = dynamic_cast<marsh::Number<double>*>(dnum);
	ASSERT_NE(nullptr, d);
	EXPECT_EQ(3.3, d->to_float64());

	delete inum;
	delete dnum;
}


TEST(PARSEATTR, NumberArr)
{
	opt::GraphInfo ginfo({});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute iattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"iarr\":{\"values\":[3,4,5]}}", &iattr, options));

	auto inums = opt::parse_attr(iattr, ginfo);
	auto i = dynamic_cast<marsh::NumArray<int64_t>*>(inums);
	ASSERT_NE(nullptr, i);
	EXPECT_EQ(3, i->size());
	EXPECT_TRUE(i->is_integral());
	std::vector<int64_t> expecti{3, 4, 5};
	EXPECT_VECEQ(expecti, i->contents_);

	query::Attribute dattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"darr\":{\"values\":[3.3,4.3,5.4]}}", &dattr, options));

	auto dnums = opt::parse_attr(dattr, ginfo);
	auto d = dynamic_cast<marsh::NumArray<double>*>(dnums);
	ASSERT_NE(nullptr, d);
	EXPECT_EQ(3, d->size());
	EXPECT_FALSE(d->is_integral());
	std::vector<double> expectd{3.3,4.3,5.4};
	EXPECT_VECEQ(expectd, d->contents_);

	delete inums;
	delete dnums;
}


TEST(PARSEATTR, String)
{
	opt::GraphInfo ginfo({});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute sattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"str\":\"hello\"}", &sattr, options));

	auto str = opt::parse_attr(sattr, ginfo);
	auto s = dynamic_cast<marsh::String*>(str);
	ASSERT_NE(nullptr, s);
	EXPECT_STREQ("hello", s->to_string().c_str());

	delete str;
}


TEST(PARSEATTR, TensorObj)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	std::vector<double> exdata{2, 8, 4, 5};
	MockDeviceRef mockref1;
	MockDeviceRef mockref2;
	MockDeviceRef mockref3;
	auto a = make_var<double>(exdata.data(), mockref1, teq::Shape({1, 4}));
	auto b = make_var<double>(exdata.data(), mockref2, teq::Shape({2, 2}));
	auto c = make_var<double>(exdata.data(), mockref3, teq::Shape({2, 2}));

	opt::GraphInfo ginfo({a, b, c});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute sattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"node\":{\"leaf\":{\"shape\":[1,4]}}}", &sattr, options));

	auto tens = opt::parse_attr(sattr, ginfo);
	auto t = dynamic_cast<teq::TensorObj*>(tens);
	ASSERT_NE(nullptr, t);
	EXPECT_EQ(a.get(), t->get_tensor().get());

	query::Attribute ambigattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"node\":{\"leaf\":{\"shape\":[2,2]}}}", &ambigattr, options));
	std::string fatalmsg = "ambiguous node attribute";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(opt::parse_attr(ambigattr, ginfo), fatalmsg.c_str());

	delete tens;
	global::set_logger(new exam::NoSupportLogger());
}


TEST(PARSEATTR, LayerObj)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

	std::vector<double> exdata{2, 8, 4, 5};
	MockDeviceRef mockref1;
	MockDeviceRef mockref2;
	MockDeviceRef mockref3;
	auto a = make_var<double>(exdata.data(), mockref1, teq::Shape({1, 4}));
	auto b = make_var<double>(exdata.data(), mockref2, teq::Shape({2, 2}));
	auto c = make_var<double>(exdata.data(), mockref3, teq::Shape({1, 4}));

	opt::GraphInfo ginfo({a, b, c});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;

	query::Attribute sattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"layer\":{\"name\":\"bigbrain\",\"input\":{\"leaf\":{\"shape\":[2,2]}}}}", &sattr, options));

	auto tens = opt::parse_attr(sattr, ginfo);
	auto t = dynamic_cast<teq::LayerObj*>(tens);
	ASSERT_NE(nullptr, t);
	EXPECT_EQ(b.get(), t->get_tensor().get());
	EXPECT_STREQ("bigbrain", t->to_string().c_str());

	query::Attribute ambigattr;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"layer\":{\"name\":\"bigbrain\",\"input\":{\"leaf\":{\"shape\":[1,4]}}}}", &ambigattr, options));
	std::string fatalmsg = "ambiguous layer attribute";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(opt::parse_attr(ambigattr, ginfo), fatalmsg.c_str());

	query::Attribute noname;
	ASSERT_EQ(google::protobuf::util::Status::OK,
		google::protobuf::util::JsonStringToMessage(
			"{\"layer\":{\"input\":{\"leaf\":{\"shape\":[2,2]}}}}", &noname, options));
	std::string fatalmsg1 = "cannot parse layer attribute unnamed or without input";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(opt::parse_attr(noname, ginfo), fatalmsg1.c_str());

	delete tens;
	global::set_logger(new exam::NoSupportLogger());
}


#endif // DISABLE_OPT_PARSEATTR_TEST
