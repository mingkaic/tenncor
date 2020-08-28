
#ifndef DISABLE_DEPEND_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/tenncor.hpp"


TEST(DEPEND, Chaining)
{
	eigen::Device device;
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

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor src2 = eteq::make_constant<double>(data2.data(), shape);
	eteq::ETensor src3 = eteq::make_constant<double>(data2.data(), shape);

	auto op = src + src2;
	auto happen_first = src * src2;

	EXPECT_FATAL(tenncor().depends(op, eteq::ETensorsT{}), "cannot depend on nothing");

	auto depped = tenncor().depends(op, {happen_first});

	auto fdep = dynamic_cast<teq::iFunctor*>(depped.get());
	ASSERT_NE(nullptr, fdep);
	auto attrs = fdep->ls_attrs();
	EXPECT_EQ(1, attrs.size());
	auto deps_attr = dynamic_cast<teq::TensArrayT*>(
		fdep->get_attr(eteq::dependency_key));
	EXPECT_NE(nullptr, deps_attr);
	EXPECT_EQ(1, deps_attr->size());

	auto children = fdep->get_args();
	ASSERT_EQ(2, children.size());

	auto depends = fdep->get_dependencies();
	ASSERT_EQ(3, depends.size());
	EXPECT_EQ(children[0].get(), depends[0].get());
	EXPECT_EQ(children[1].get(), depends[1].get());
	EXPECT_EQ(happen_first.get(), depends[2].get());

	eteq::EVariable<double> buffer = eteq::make_variable<double>(data.data(), shape);
	auto ass = tenncor().assign(buffer, happen_first);

	static_cast<teq::iFunctor*>(op.get())->update_child(buffer, 0);
	// assign replaces happen_first
	fdep->update_child(ass, 2);

	// assign is an indirect dependency of op
	teq::Evaluator eval;
	eval.evaluate(device, {depped.get()});
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
