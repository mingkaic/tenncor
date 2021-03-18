
#ifndef DISABLE_UTILS_SPARSE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/utils/sparse/random.hpp"


static global::GenPtrT test_generator = std::make_shared<global::Randomizer>();


TEST(SPARSE_UTIL, Shuffle)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	std::vector<size_t> indices(15, 0);
	auto it = indices.begin();
	auto et = indices.end();
	std::iota(it, et, 0);
	eigen::fisher_yates_shuffle(it, et, 8, test_generator);

	std::vector<size_t> expect = {
		0, 4, 9, 1, 7, 3, 6, 8, 5, 2, 10, 11, 12, 13, 14
	};
	EXPECT_ARREQ(expect, indices);
}


TEST(SPARSE_UTIL, BigShuffle)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	std::vector<size_t> indices(10000, 0);
	auto it = indices.begin();
	auto et = indices.end();
	std::iota(it, et, 0);
	eigen::fisher_yates_shuffle(it, et, 5062, test_generator);
	EXPECT_EQ(8214, indices[5061]);
}


TEST(SPARSE_UTIL, RandIndicesBaseCase)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	std::vector<size_t> base(10);
	auto it = base.begin();
	auto et = base.end();
	eigen::random_indices(it, et, 1000, test_generator);

	std::vector<size_t> expect = {
		6, 231, 486, 923, 217, 172, 641, 98, 445, 763
	};
	EXPECT_VECEQ(expect, base);
}


TEST(SPARSE_UTIL, RandIndices)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	std::vector<size_t> lcase(10);
	auto it = lcase.begin();
	auto et = lcase.end();
	eigen::random_indices(it, et, 3211223812, test_generator);

	std::vector<size_t> expect = {
		628017, 823025331, 1748437723, 392026577, 1021387693,
		416737602, 2663579196, 909171585, 2143717325, 575434738
	};
	EXPECT_VECEQ(expect, lcase);
}


TEST(SPARSE_UTIL, RealRandIndices)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	teq::Shape shape({45000, 45000});
	float density = 0.00005;
	size_t nelems = shape.n_elems();
	size_t nzs = density * nelems;

	std::vector<size_t> sparse(nzs);
	auto it = sparse.begin();
	auto et = sparse.end();
	eigen::random_indices(it, et, nelems, test_generator);

	EXPECT_EQ(2011009589, sparse.back());
}


TEST(SPARSE_UTIL, RandSparse)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	teq::Shape shape({45000, 45000});
	eigen::TripletsT<float> trips;
	float density = 0.00005;
	eigen::random_sparse<float>(trips, shape, density, test_generator);

	auto& last = trips.back();
	EXPECT_EQ(last.col(), 4589);
	EXPECT_EQ(last.row(), 44689);
	EXPECT_DOUBLE_EQ(last.value(), -0.94800132513046265);
}


#endif // DISABLE_UTILS_SPARSE_TEST
