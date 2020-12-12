
#ifndef DISABLE_EIGEN_TYPER_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/eigen/mock/mock.hpp"


using ::testing::_;
using ::testing::An;
using ::testing::Return;
using ::testing::Throw;


struct TYPER : public tutil::TestcaseWithLogger<> {};


TEST_F(TYPER, Default)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::TypeParser<egen::ADD> parser;
	marsh::Maps attrs;

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(attrs, {}), eigen::no_argument_err.c_str());

	// deduce to type of highest precision, which is double
	// float -> double
	EXPECT_EQ(egen::DOUBLE, parser(attrs, eigen::DTypesT{egen::DOUBLE, egen::FLOAT}));
	// ensure order doesn't matter
	EXPECT_EQ(egen::DOUBLE, parser(attrs, eigen::DTypesT{egen::FLOAT, egen::DOUBLE}));

	// int32_t -> float
	EXPECT_EQ(egen::FLOAT, parser(attrs, eigen::DTypesT{egen::FLOAT, egen::INT32}));
	// ensure order doesn't matter
	EXPECT_EQ(egen::FLOAT, parser(attrs, eigen::DTypesT{egen::INT32, egen::FLOAT}));
}


TEST_F(TYPER, Assign)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::TypeParser<egen::ASSIGN> parser;
	marsh::Maps attrs;

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(attrs, {}), eigen::no_argument_err.c_str());

	// assign always takes the first type (type of destination variable)
	EXPECT_EQ(egen::DOUBLE, parser(attrs, eigen::DTypesT{egen::DOUBLE, egen::FLOAT}));
	EXPECT_EQ(egen::FLOAT, parser(attrs, eigen::DTypesT{egen::FLOAT, egen::DOUBLE}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32, egen::DOUBLE}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32, egen::FLOAT}));
}


TEST_F(TYPER, Cast)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::TypeParser<egen::CAST> parser;
	marsh::Maps attrs;

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(attrs, {}), eigen::no_argument_err.c_str());

	// without dtype attribute, just identity cast
	EXPECT_EQ(egen::DOUBLE, parser(attrs, eigen::DTypesT{egen::DOUBLE}));
	EXPECT_EQ(egen::FLOAT, parser(attrs, eigen::DTypesT{egen::FLOAT}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32}));

	eigen::Packer<egen::_GENERATED_DTYPE> packer;
	packer.pack(attrs, egen::INT32);
	// with dtype attribute, cast to attribute type
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::DOUBLE}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::FLOAT}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32}));
}


#endif // DISABLE_EIGEN_TYPER_TEST
