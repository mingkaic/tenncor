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
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape cshape = clist;
	clay::Shape pshape = make_partial(this, clist);

	kiln::Validator valid(pshape, {});

	// default
	kiln::ConstInit ci;
	kiln::ConstInit vci;

	// validated
	kiln::ConstInit vi(valid);

	double cdata = get_double(1, "ci")[0];
	std::vector<double> vcdata = get_double(get_int(1, "vci.size", {23, 52})[0], "vci");
	ci.set<double>(cdata);
	vci.set<double>(vcdata);
	vi.set<double>(cdata);

	// string init
	kiln::ConstInit sci(std::string((char*) &cdata, sizeof(double)), clay::DTYPE::DOUBLE);
	kiln::ConstInit svci(std::string((char*) &vcdata[0], vcdata.size() * sizeof(double)), clay::DTYPE::DOUBLE);

	clay::TensorPtrT cten = ci.get(cshape);
	ASSERT_NE(nullptr, cten);
	clay::TensorPtrT vcten = vci.get(cshape);
	ASSERT_NE(nullptr, vcten);
	clay::TensorPtrT vten = vi.get(cshape);
	ASSERT_NE(nullptr, vten);

	ASSERT_SHAPEQ(cshape, cten->get_shape());
	ASSERT_SHAPEQ(cshape, vcten->get_shape());
	ASSERT_SHAPEQ(cshape, vten->get_shape());

	double* cd = (double*) cten->get_state().data_.lock().get();
	double* vcd = (double*) vcten->get_state().data_.lock().get();
	double* vd = (double*) vten->get_state().data_.lock().get();

	size_t n = cshape.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		ASSERT_EQ(cdata, cd[i])
			<< "failed at " << i;
		ASSERT_EQ(cdata, vd[i])
			<< "failed at " << i;
		ASSERT_EQ(vcdata[i % vcdata.size()], vcd[i])
			<< "failed at " << i;
	}
}


TEST_F(SPEC_INIT, UnifInit_B001)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape cshape = clist;
	clay::Shape pshape = make_partial(this, clist);

	kiln::Validator valid(pshape, {});

	// default
	kiln::UnifInit ui;

	// validated
	kiln::UnifInit vi(valid);

	std::vector<double> udata = get_double(2, "ui", {-12, 24});
	double umin = std::min(udata[0], udata[1]);
	double umax = std::max(udata[0], udata[1]);
	ui.set<double>(umin, umax);
	vi.set<double>(umin, umax);

	// string init
	kiln::UnifInit sui(
		std::string((char*) &umin, sizeof(double)),
		std::string((char*) &umax, sizeof(double)), clay::DTYPE::DOUBLE);

	clay::TensorPtrT uten = ui.get(cshape);
	ASSERT_NE(nullptr, uten);
	clay::TensorPtrT vten = vi.get(cshape);
	ASSERT_NE(nullptr, vten);

	ASSERT_SHAPEQ(cshape, uten->get_shape());
	ASSERT_SHAPEQ(cshape, vten->get_shape());

	double* ud = (double*) uten->get_state().data_.lock().get();
	double* vd = (double*) vten->get_state().data_.lock().get();

	size_t n = cshape.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		ASSERT_GE(umax, ud[i])
			<< "failed at " << i;
		ASSERT_LE(umin, ud[i])
			<< "failed at " << i;
		ASSERT_GE(umax, vd[i])
			<< "failed at " << i;
		ASSERT_LE(umin, vd[i])
			<< "failed at " << i;
	}
}


// todo: re-enable once shape gen of high minimum n is fixed
TEST_F(SPEC_INIT, DISABLED_NormInit_B001)
{
	std::vector<size_t> clist = random_def_shape(this, {1, 6}, {3562, 8002});
	clay::Shape cshape = clist;
	clay::Shape pshape = make_partial(this, clist);

	kiln::Validator valid(pshape, {});

	// default
	kiln::NormInit ni;

	// validated
	kiln::NormInit vi(valid);

	std::vector<double> ndata = get_double(2, "ni", {0.5, 2});
	double nmean = ndata[0];
	double nstdev = ndata[1];
	ni.set<double>(nmean, nstdev);
	vi.set<double>(nmean, nstdev);

	// string init
	kiln::NormInit sni(
		std::string((char*) &nmean, sizeof(double)),
		std::string((char*) &nstdev, sizeof(double)),clay::DTYPE::DOUBLE);

	clay::TensorPtrT nten = ni.get(cshape);
	ASSERT_NE(nullptr, nten);
	clay::TensorPtrT vten = vi.get(cshape);
	ASSERT_NE(nullptr, vten);

	ASSERT_SHAPEQ(cshape, nten->get_shape());
	ASSERT_SHAPEQ(cshape, vten->get_shape());

	double* nd = (double*) nten->get_state().data_.lock().get();
	double* vd = (double*) vten->get_state().data_.lock().get();

	size_t n = cshape.n_elems();
	std::vector<size_t> stdev_count(3, 0);
	std::vector<size_t> vstdev_count(3, 0);
	for (size_t i = 0; i < n; ++i)
	{
		size_t ni = std::abs(nmean - nd[i]) / nstdev;
		size_t vi = std::abs(nmean - vd[i]) / nstdev;

		if (ni < 3)
		{
			stdev_count[ni]++;
		}
		if (vi < 3)
		{
			vstdev_count[vi]++;
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

	float vexpect68 = (float) vstdev_count[0] / n; // expect ~68%
	float vexpect95 = (float) (vstdev_count[0] + vstdev_count[1]) / n; // expect ~95%
	float vexpect99 = (float) (vstdev_count[0] + vstdev_count[1] + vstdev_count[2]) / n; // expect ~99.7%

	float verr1 = std::abs(0.68 - vexpect68);
	float verr2 = std::abs(0.95 - vexpect95);
	float verr3 = std::abs(0.997 - vexpect99);

	EXPECT_GT(ERR_THRESH, verr1);
	EXPECT_GT(ERR_THRESH, verr2);
	EXPECT_GT(ERR_THRESH, verr3);
}


#endif /* DISABLE_SPEC_INIT_TEST */


#endif /* DISABLE_KILN_MODULE_TESTS */
