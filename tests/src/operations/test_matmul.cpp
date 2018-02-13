//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATION_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "include/operations/operations.hpp"
#include "include/utils/futils.hpp"

#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"


#ifndef DISABLE_MATMUL_TEST


class MATMUL : public FUZZ::fuzz_test {};


using namespace nnet;


using TWODV = std::vector<std::vector<signed> >;


TWODV create2D (std::vector<signed> juanD, tensorshape mats, bool transpose = false)
{
	std::vector<size_t> dims = mats.as_list();
	size_t C = dims[0];
	size_t R = dims[1];
	TWODV res;

	size_t resC = transpose ? R : C;
	size_t resR = transpose ? C : R;
 	for (size_t y = 0; y < resR; y++)
	{
		res.push_back(std::vector<signed>(resC, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			size_t juan_coord = x + y * C;
			if (transpose)
			{
				res[x][y] = juanD[juan_coord];
			}
			else
			{
				res[y][x] = juanD[juan_coord];
			}
		}
	}
	return res;
}


bool freivald (FUZZ::fuzz_test* fuzzer, TWODV a, TWODV b, TWODV c)
{
	assert(!b.empty());
	size_t rlen = b[0].size();
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	size_t m = 10;
	for (size_t i = 0; i < m; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<size_t> r = fuzzer->get_int(rlen, nnutils::formatter() << "freivald_vec" << i, {0, 1});

		// p = a @ (b @ r) - c @ r
		std::vector<signed> br;
		for (size_t y = 0, n = b.size(); y < n; y++)
		{
			signed bri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<signed> cr;
		for (size_t y = 0, n = c.size(); y < n; y++)
		{
			signed cri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<signed> p;
		size_t n = a.size();
		for (size_t y = 0; y < n; y++)
		{
			signed ari = 0;
			for (size_t x = 0, m = a[y].size(); x < m; x++)
			{
				ari += a[y][x] * br[x];
			}
			p.push_back(ari);
		}
		for (size_t j = 0; j < n; j++)
		{
			p[j] -= cr[j];
		}

		// if p != 0 -> return false
		if (!std::all_of(p.begin(), p.end(), [](signed d) { return d == 0; }))
			return false;
	}
	return true;
}


TEST_F(MATMUL, NullptrRet_C000)
{
	variable* zero = new variable(0);
	EXPECT_EQ(nullptr, matmul(nullptr, nullptr));
	EXPECT_EQ(nullptr, matmul(zero, nullptr));
	EXPECT_EQ(nullptr, matmul(nullptr, zero));
	delete zero;
}


TEST_F(MATMUL, Matmul_C001)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform_int rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::INT, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::INT, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::INT, "tA");
	variable tB(shapetB, rinit, nnet::INT, "tB");

	// shapes of <k, n>
	varptr res = matmul(varptr(&A), varptr(&B));
	varptr restA = matmul(varptr(&tA), varptr(&B), true);
	varptr restB = matmul(varptr(&A), varptr(&tB), false, true);
	varptr resT = matmul(varptr(&tA), varptr(&tB), true, true);

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape));

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);

	// Freivald's algorithm
	EXPECT_TRUE(freivald(this, matA, matB, matres));
	EXPECT_TRUE(freivald(this, mattA, matB, matrestA));
	EXPECT_TRUE(freivald(this, matA, mattB, matrestB));
	EXPECT_TRUE(freivald(this, mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


// tests matrix multiplication but for n dimensions, matrix sizes reduced to 2-5, (we get at most 5x25 matmuls)
// todo: test
TEST_F(MATMUL, DISABLED_NDim_Matmul_C001)
{
}


TEST_F(MATMUL, Incompatible_C002)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]+1};

	variable A(shapeA, rinit, nnet::DOUBLE, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::DOUBLE, "B"); // shape <k, m+1>

	A.initialize();
	B.initialize();

	varptr bad = matmul(varptr(&A), varptr(&B));
	EXPECT_THROW(bad->eval(), std::logic_error);
}


TEST_F(MATMUL, Jacobian_C003)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform rinit(0, 1);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::DOUBLE, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::DOUBLE, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::DOUBLE, "tA");
	variable tB(shapetB, rinit, nnet::DOUBLE, "tB");

	// shapes of <k, n>
	varptr res = sigmoid(varptr(matmul(varptr(&A), varptr(&B))));
	varptr restA = sigmoid(varptr(matmul(varptr(&tA), varptr(&B), true)));
	varptr restB = sigmoid(varptr(matmul(varptr(&A), varptr(&tB), false, true)));
	varptr resT = sigmoid(varptr(matmul(varptr(&tA), varptr(&tB), true, true)));

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	inode* dresA = res->derive(&A);
	inode* dresB = res->derive(&B);

	inode* drestAA = restA->derive(&tA);
	inode* drestAB = restA->derive(&B);

	inode* drestBA = restB->derive(&A);
	inode* drestBB = restB->derive(&tB);

	inode* dresTA = resT->derive(&tA);
	inode* dresTB = resT->derive(&tB);

	// requires on all elementary operations to be valid (not a great validation method...)
	// res = 1/(1+e^-(A@B))
	// dres = jacobian(sigmoid'(1))
	// where jacobian = {
	// 		sigmoid'(1) @ B^T for dA
	//		A^T @ sigmoid'(1) for dB
	// }
	// sigmoid' = sigmoid * (1 - sigmoid)
	varptr dsig_res = res * (1.0 - res);
	inode* fake_dresA = matmul(dsig_res, &B, false, true);
	inode* fake_dresB = matmul(&A, dsig_res, true);

	varptr dsig_restA = restA * (1.0 - restA);
	inode* fake_drestAA = transpose(matmul(dsig_restA, &B, false, true));
	inode* fake_drestAB = matmul(&tA, dsig_restA);

	varptr dsig_restB = restB * (1.0 - restB);
	inode* fake_drestBA = matmul(dsig_restB, &tB);
	inode* fake_drestBB = transpose(matmul(&A, dsig_restB, true, false));

	varptr dsig_resT = resT * (1.0 - resT);
	inode* fake_dresTA = transpose(matmul(dsig_resT, &tB));
	inode* fake_dresTB = transpose(matmul(&tA, dsig_resT));

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), A.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), B.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), tA.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), B.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), A.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), tB.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), tA.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), tB.get_shape()));

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), fake_dresA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), fake_dresB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), fake_drestAA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), fake_drestAB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), fake_drestBA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), fake_drestBB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), fake_dresTA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), fake_dresTB->get_shape()));

	std::vector<double> dresA_data = expose<double>(dresA);
	std::vector<double> dresB_data = expose<double>(dresB);
	std::vector<double> drestAA_data = expose<double>(drestAA);
	std::vector<double> drestAB_data = expose<double>(drestAB);
	std::vector<double> drestBA_data = expose<double>(drestBA);
	std::vector<double> drestBB_data = expose<double>(drestBB);
	std::vector<double> dresTA_data = expose<double>(dresTA);
	std::vector<double> dresTB_data = expose<double>(dresTB);

	std::vector<double> fake_dresA_data = expose<double>(fake_dresA);
	std::vector<double> fake_dresB_data = expose<double>(fake_dresB);
	std::vector<double> fake_drestAA_data = expose<double>(fake_drestAA);
	std::vector<double> fake_drestAB_data = expose<double>(fake_drestAB);
	std::vector<double> fake_drestBA_data = expose<double>(fake_drestBA);
	std::vector<double> fake_drestBB_data = expose<double>(fake_drestBB);
	std::vector<double> fake_dresTA_data = expose<double>(fake_dresTA);
	std::vector<double> fake_dresTB_data = expose<double>(fake_dresTB);

	// all a shapes should have the same number of elements
	double err_thresh = 0.0000001;
	for (size_t i = 0, n = dresA_data.size(); i < n; i++)
	{
		double dresAerr = std::abs(dresA_data[i] - fake_dresA_data[i]);
		double drestAAerr = std::abs(drestAA_data[i] - fake_drestAA_data[i]);
		double drestBAerr = std::abs(drestBA_data[i] - fake_drestBA_data[i]);
		double dresTAerr = std::abs(dresTA_data[i] - fake_dresTA_data[i]);
		EXPECT_GT(err_thresh, dresAerr);
		EXPECT_GT(err_thresh, drestAAerr);
		EXPECT_GT(err_thresh, drestBAerr);
		EXPECT_GT(err_thresh, dresTAerr);
	}
	for (size_t i = 0, n = dresB_data.size(); i < n; i++)
	{
		double dresBerr = std::abs(dresB_data[i] - fake_dresB_data[i]);
		double drestABerr = std::abs(drestAB_data[i] - fake_drestAB_data[i]);
		double drestBBerr = std::abs(drestBB_data[i] - fake_drestBB_data[i]);
		double dresTBerr = std::abs(dresTB_data[i] - fake_dresTB_data[i]);
		EXPECT_GT(err_thresh, dresBerr);
		EXPECT_GT(err_thresh, drestABerr);
		EXPECT_GT(err_thresh, drestBBerr);
		EXPECT_GT(err_thresh, dresTBerr);
	}
}


// tests large matrices sizes (100-112), 2D only
TEST_F(MATMUL, Strassen_C004)
{
	// we get at most 12996 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {STRASSEN_THRESHOLD, STRASSEN_THRESHOLD+12});
	rand_uniform_int rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::INT, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::INT, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::INT, "tA");
	variable tB(shapetB, rinit, nnet::INT, "tB");

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	// shapes of <k, n>
//	clock_t t = clock();
	varptr res = matmul(varptr(&A), varptr(&B));
//	const double work_time1 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr restA = matmul(varptr(&tA), varptr(&B), true);
//	const double work_time2 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr restB = matmul(varptr(&A), varptr(&tB), false, true);
//	const double work_time3 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr resT = matmul(varptr(&tA), varptr(&tB), true, true);
//	const double work_time4 = (clock() - t) / double(CLOCKS_PER_SEC);
//	ASSERT_GT(0.3, work_time1);
//	ASSERT_GT(0.3, work_time2);
//	ASSERT_GT(0.3, work_time3);
//	ASSERT_GT(0.3, work_time4);

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape));

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);
	// Freivald's algorithm

	EXPECT_TRUE(freivald(this, matA, matB, matres));
	EXPECT_TRUE(freivald(this, mattA, matB, matrestA));
	EXPECT_TRUE(freivald(this, matA, mattB, matrestB));
	EXPECT_TRUE(freivald(this, mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


#endif /* DISABLE_MATMUL_TEST */



#endif /* DISABLE_OPERATION_MODULE_TESTS */

