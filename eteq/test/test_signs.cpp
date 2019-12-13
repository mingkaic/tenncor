
#ifndef DISABLE_SIGNS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/generated/api.hpp"

#include "eteq/placeholder.hpp"


TEST(PLACEHOLDER, AssignData)
{
	teq::ShapeSignature sign({3, 2});
	teq::Shape shape(sign);
	eteq::PlaceLink<double> ph(sign, "ph");

	EXPECT_FALSE(ph.can_build());
	EXPECT_FATAL(ph.build_data(),
		"cannot get tensor from unassigned placeholder");

	eigen::TensorT<double> data(3, 2, 1, 1, 1, 1, 1, 1);
	data.setRandom();
	ph.assign(data);

	EXPECT_TRUE(ph.can_build());
	auto tens = ph.build_data();
	auto var = dynamic_cast<eteq::Variable<double>*>(tens.get());
	EXPECT_NE(nullptr, var);
}


TEST(PLACEHOLDER, AssignTens)
{
	teq::ShapeSignature sign({3, 2});
	teq::Shape shape(sign);
	eteq::PlaceLink<double> ph(sign, "ph");

	EXPECT_FALSE(ph.can_build());
	EXPECT_FATAL(ph.build_data(),
		"cannot get tensor from unassigned placeholder");

	auto var = eteq::make_variable_scalar<double>(0., shape, "var");
	ph.assign(var);

	EXPECT_TRUE(ph.can_build());
	auto tens = ph.build_data();
	EXPECT_EQ(var.get(), tens.get());
}


TEST(PLACEHOLDER, AssignDataSigned)
{
	teq::ShapeSignature sign({3, 0});
	eteq::PlaceLink<double> ph(sign, "ph");

	EXPECT_FALSE(ph.can_build());
	EXPECT_FATAL(ph.build_data(),
		"cannot get tensor from unassigned placeholder");

	eigen::TensorT<double> data(3, 2, 1, 1, 1, 1, 1, 1);
	data.setRandom();
	ph.assign(data);

	EXPECT_TRUE(ph.can_build());
	auto tens = ph.build_data();
	auto var = dynamic_cast<eteq::Variable<double>*>(tens.get());
	EXPECT_NE(nullptr, var);
}


TEST(PLACEHOLDER, AssignTensSigned)
{
	teq::ShapeSignature sign({3, 0});
	eteq::PlaceLink<double> ph(sign, "ph");

	EXPECT_FALSE(ph.can_build());
	EXPECT_FATAL(ph.build_data(),
		"cannot get tensor from unassigned placeholder");

	teq::Shape shape({3, 3});
	auto var = eteq::make_variable_scalar<double>(0., shape, "var");
	ph.assign(var);

	EXPECT_TRUE(ph.can_build());
	auto tens = ph.build_data();
	EXPECT_EQ(var.get(), tens.get());
}


TEST(FUNCSIGN, UseSignature)
{
	teq::ShapeSignature sign({3, 2});
	auto ph = std::make_shared<eteq::PlaceLink<double>>(sign, "left");

	teq::Shape varshape({3, 3});
	auto var = eteq::make_variable_scalar<double>(0., varshape, "right");

	auto mat = tenncor::matmul(eteq::LinkptrT<double>(ph), eteq::to_link<double>(var));

	EXPECT_FALSE(mat->can_build());
	EXPECT_FATAL(mat->build_data(), "cannot get data from unbuildable signature MATMUL");

	eigen::TensorT<double> data(3, 2, 1, 1, 1, 1, 1, 1);
	data.setRandom();
	ph->assign(data);

	EXPECT_TRUE(mat->can_build());
	auto tens = mat->build_data();
	auto func = dynamic_cast<eteq::Functor<double>*>(tens.get());
	ASSERT_NE(nullptr, func);
	auto shape = func->shape();
	teq::Shape expect_shape(sign);
	EXPECT_ARREQ(expect_shape, shape);
}


TEST(FUNCSIGN, UseSignatureSigned)
{
	teq::ShapeSignature sign({3, 0});
	auto ph = std::make_shared<eteq::PlaceLink<double>>(sign, "left");

	teq::Shape varshape({3, 3});
	auto var = eteq::make_variable_scalar<double>(0., varshape, "right");

	auto mat = tenncor::matmul(eteq::LinkptrT<double>(ph), eteq::to_link<double>(var));

	EXPECT_FALSE(mat->can_build());
	EXPECT_FATAL(mat->build_data(), "cannot get data from unbuildable signature MATMUL");

	eigen::TensorT<double> data(3, 4, 1, 1, 1, 1, 1, 1);
	data.setRandom();
	ph->assign(data);

	EXPECT_TRUE(mat->can_build());
	auto tens = mat->build_data();
	auto func = dynamic_cast<eteq::Functor<double>*>(tens.get());
	ASSERT_NE(nullptr, func);
	auto shape = func->shape();
	teq::Shape expect_shape({3, 4});
	EXPECT_ARREQ(expect_shape, shape);
}


#endif // DISABLE_SIGNS_TEST
