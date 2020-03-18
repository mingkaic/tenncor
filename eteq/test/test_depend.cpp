
#ifndef DISABLE_DEPEND_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/depend.hpp"

#include "generated/api.hpp"


TEST(DEPEND, Chaining)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> src2 = eteq::make_constant<double>(data2.data(), shape);
	eteq::ETensor<double> src3 = eteq::make_constant<double>(data2.data(), shape);

	auto op = src + src2;
	auto happen_first = src * src2;

	EXPECT_FATAL(tenncor::depends(op, eteq::ETensorsT<double>{}), "cannot depend on nothing");

	auto depped = tenncor::depends(op, {happen_first});

	auto fdep = dynamic_cast<teq::iFunctor*>(depped.get());
	ASSERT_NE(nullptr, fdep);
	auto attrs = fdep->ls_attrs();
	EXPECT_EQ(0, attrs.size());
	EXPECT_EQ(nullptr, fdep->get_attr("anything"));
	{
		auto num = std::make_unique<marsh::Number<double>>(3);
		fdep->add_attr("anything", std::move(num));
	}
	EXPECT_NE(nullptr, fdep->get_attr("anything"));
	fdep->rm_attr("anything");
	EXPECT_EQ(nullptr, fdep->get_attr("anything"));

	auto children = fdep->get_children();
	ASSERT_EQ(4, children.size());
	EXPECT_EQ(op.get(), children[0].get());
	EXPECT_EQ(src.get(), children[1].get());
	EXPECT_EQ(src2.get(), children[2].get());
	EXPECT_EQ(happen_first.get(), children[3].get());

	EXPECT_FATAL(fdep->update_child(src3, 0), "cannot reassign non-observable dependee of depend (index 0)");

	eteq::EVariable<double> buffer = eteq::make_variable<double>(data.data(), shape);
	auto ass = tenncor::assign(buffer, happen_first);

	fdep->update_child(buffer, 1);
	fdep->update_child(ass, 3); // assign replaces happen_first

	// assign is an indirect dependency of op
	auto session = eigen::get_session();
	session.track({depped});
	session.update();
	{
		auto gotshape = depped->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) depped->device().data();
	// expect = data1 * data2 + data2
	for (size_t i = 0; i < n; ++i)
	{
		double expect = data[i] * data2[i] + data2[i];
		EXPECT_DOUBLE_EQ(expect, optr[i]);
		// otherwise if optr = data[i] + data2[i], then ass didn't go first
	}
}


#endif // DISABLE_DEPEND_TEST
