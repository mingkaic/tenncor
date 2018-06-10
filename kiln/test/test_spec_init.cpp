//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_KILN_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "kiln/const_init.hpp"
#include "kiln/unif_init.hpp"
#include "kiln/norm_init.hpp"


#ifndef DISABLE_SPEC_INIT_TEST


static const double ERR_THRESH = 0.08; // 8% error


class SPEC_INIT : public testutil::fuzz_test {};


using namespace testutil;


TEST_F(SPEC_INIT, ConstInit_B000)
{
	clay::Shape cshape = random_def_shape(this);
	double cdata = get_double(1, "ci")[0];
	std::vector<double> vdata = get_double(get_int(1, "vci.size", {23, 52})[0], "vci");
	clay::BuildTensorF ci = kiln::const_init(cdata, cshape);
	clay::BuildTensorF vi = kiln::const_init(vdata, cshape);

	clay::TensorPtrT cten = ci();
	ASSERT_NE(nullptr, cten);
	clay::TensorPtrT vten = vi();
	ASSERT_NE(nullptr, vten);

	ASSERT_SHAPEQ(cshape, cten->get_shape());
	ASSERT_SHAPEQ(cshape, vten->get_shape());

	double* cd = (double*) cten->get_state().get();
	double* vd = (double*) vten->get_state().get();

	size_t n = cshape.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		ASSERT_EQ(cdata, cd[i])
			<< "failed at " << i;
		ASSERT_EQ(vdata[i % vdata.size()], vd[i])
			<< "failed at " << i;
	}
}


TEST_F(SPEC_INIT, UnifInit_B001)
{
	clay::Shape cshape = random_def_shape(this);
	std::vector<double> udata = get_double(2, "ui", {-12, 24});
	double umin = std::min(udata[0], udata[1]);
	double umax = std::max(udata[0], udata[1]);
	clay::BuildTensorF ui = kiln::unif_init(umin, umax, cshape);

	clay::TensorPtrT uten = ui();
	ASSERT_NE(nullptr, uten);
	ASSERT_SHAPEQ(cshape, uten->get_shape());
	double* ud = (double*) uten->get_state().get();

	size_t n = cshape.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		ASSERT_GE(umax, ud[i])
			<< "failed at " << i;
		ASSERT_LE(umin, ud[i])
			<< "failed at " << i;
	}
}


// todo: re-enable once shape gen of high minimum n is fixed
TEST_F(SPEC_INIT, DISABLED_NormInit_B001)
{
	clay::Shape cshape = random_def_shape(this, {1, 6}, {3562, 8002});
	std::vector<double> ndata = get_double(2, "ni", {0.5, 2});
	double nmean = ndata[0];
	double nstdev = ndata[1];
	clay::BuildTensorF ni = kiln::norm_init(nmean, nstdev, cshape);

	clay::TensorPtrT nten = ni();
	ASSERT_NE(nullptr, nten);
	ASSERT_SHAPEQ(cshape, nten->get_shape());
	double* nd = (double*) nten->get_state().get();

	size_t n = cshape.n_elems();
	std::vector<size_t> stdev_count(3, 0);
	for (size_t i = 0; i < n; ++i)
	{
		size_t ni = std::abs(nmean - nd[i]) / nstdev;

		if (ni < 3)
		{
			stdev_count[ni]++;
		}
	}
	// check the first 3 stdev
	float expect68 = (float) stdev_count[0] / n; // expect ~68%
	float expect95 = (float) (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	float expect99 = (float) (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	float err1 = std::abs(0.68 - expect68);
	float err2 = std::abs(0.95 - expect95);
	float err3 = std::abs(0.997 - expect99);

	EXPECT_GT(ERR_THRESH, err1);
	EXPECT_GT(ERR_THRESH, err2);
	EXPECT_GT(ERR_THRESH, err3);
}


#endif /* DISABLE_SPEC_INIT_TEST */


#endif /* DISABLE_KILN_MODULE_TESTS */
