#include "gtest/gtest.h"

#include "sand/operator.hpp"
#include "sand/include/unary.hpp"
#include "sand/include/binary.hpp"
#include "sand/include/matmul.hpp"


#ifndef DISABLE_OPERATOR_TEST


class OPERATOR : public ::testing::Test {};


TEST_F(OPERATOR, Abs)
{}


TEST_F(OPERATOR, Neg)
{}


TEST_F(OPERATOR, Sin)
{}


TEST_F(OPERATOR, Cos)
{}


TEST_F(OPERATOR, Tan)
{}


TEST_F(OPERATOR, Exp)
{}


TEST_F(OPERATOR, Log)
{}


TEST_F(OPERATOR, Sqrt)
{}


TEST_F(OPERATOR, Pow)
{}


TEST_F(OPERATOR, Add)
{}


TEST_F(OPERATOR, Sub)
{}


TEST_F(OPERATOR, Mul)
{}


TEST_F(OPERATOR, Div)
{}


TEST_F(OPERATOR, Matmul)
{}


#endif /* DISABLE_OPERATOR_TEST */
