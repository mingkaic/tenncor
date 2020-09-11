
#ifndef DISABLE_EIGEN_TYPER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mutable_leaf.hpp"
#include "internal/eigen/eigen.hpp"


TEST(TYPER, Default)
{
	egen::TypeParser<egen::ADD> parser;
	marsh::Maps attrs;

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


TEST(TYPER, Assign)
{
	egen::TypeParser<egen::ASSIGN> parser;
	marsh::Maps attrs;

	EXPECT_FATAL(parser(attrs, {}), eigen::no_argument_err.c_str());

	// assign always takes the first type (type of destination variable)
	EXPECT_EQ(egen::DOUBLE, parser(attrs, eigen::DTypesT{egen::DOUBLE, egen::FLOAT}));
	EXPECT_EQ(egen::FLOAT, parser(attrs, eigen::DTypesT{egen::FLOAT, egen::DOUBLE}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32, egen::DOUBLE}));
	EXPECT_EQ(egen::INT32, parser(attrs, eigen::DTypesT{egen::INT32, egen::FLOAT}));
}


TEST(TYPER, Cast)
{
	egen::TypeParser<egen::CAST> parser;
	marsh::Maps attrs;

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
