
#ifndef DISABLE_HONE_MATCHAIN_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/utils/sparse/random.hpp"

#include "tenncor/hone/hone.hpp"


static global::GenPtrT test_generator = std::make_shared<global::Randomizer>();


template <typename T>
teq::TensptrT make_randsparse (const teq::Shape& shape, float density,
	std::string label)
{
	if (density == 1.f) // not sparse
	{
		eigen::MatrixT<T> m(shape.at(1), shape.at(0));
		m.setRandom();
		return eteq::make_variable<T>(m.data(), shape, label);
	}
	eigen::TripletsT<T> trips;
	eigen::random_sparse<T>(trips, shape, density, test_generator);
	eigen::SMatrixT<T> sm(shape.at(1), shape.at(0));
	sm.setFromTriplets(trips.begin(), trips.end());
	return eteq::make_variable<T>(sm.valuePtr(),
		eigen::SparseInfo::get<T>(sm), shape, label);
}


static teq::TensptrT make_sparse (teq::Shape shape, bool random)
{
	eigen::TripletsT<float> trips;
	if (random)
	{
		float density;
		if (shape.n_elems() > 10000)
		{
			density = 0.005;
		}
		else if (shape.n_elems() > 1000)
		{
			density = 0.01;
		}
		else
		{
			density = 0.1;
		}
		eigen::random_sparse<float>(
			trips, shape, density, test_generator);
	}
	else
	{
		auto gen = test_generator->unif_decgen(-1, 1);
		eigen::identity_sparse<float>(trips, shape,
		[&gen]() -> float { return gen(); });
	}
	eigen::SMatrixT<float> sm(shape.at(1), shape.at(0));
	sm.setFromTriplets(trips.begin(), trips.end());
	return eteq::make_variable<float>(sm.valuePtr(),
		eigen::SparseInfo::get<float>(sm), shape, "9");
}


TEST(MATCHAIN, SharedNodeFlattenChain)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(
		test_generator.get())->seed(seed);

	size_t inject_begin = 3;
	size_t inject_end = 7;
	teq::ShapesT outershapes = {
		teq::Shape({150, 150}), teq::Shape({4500, 150}),
		teq::Shape({4500, 4500}), teq::Shape({300, 4500}),
		teq::Shape({300, 300}), teq::Shape({9000, 300}),
		teq::Shape({9000, 9000}), teq::Shape({9000, 9000}),
		teq::Shape({9000, 9000})
	};
	teq::TensptrsT vars;
	vars.reserve(outershapes.size());
	for (auto shape : outershapes)
	{
		vars.push_back(make_sparse(shape, true));
	}

	auto vit = vars.begin();
	teq::TensptrsT expect_innerchain(vit + inject_begin, vit + inject_end);
	auto inner = expect_innerchain.front();
	for (auto it = expect_innerchain.begin() + 1, et = expect_innerchain.end();
		it != et; ++it)
	{
		inner = eteq::make_functor(egen::MATMUL, { inner, *it });
	}

	teq::TensptrsT expect_origchain(vit, vit + inject_begin);
	expect_origchain.push_back(inner);
	expect_origchain.insert(expect_origchain.end(),
		vit + inject_end, vars.end());
	auto original = expect_origchain.front();
	for (auto it = expect_origchain.begin() + 1, et = expect_origchain.end();
		it != et; ++it)
	{
		original = eteq::make_functor(egen::MATMUL, { original, *it });
	}

	// expect original shape [9000,4500]
	teq::TensptrsT expect_orig2chain = {
		inner, make_sparse(teq::Shape({100, 9000}), true)
	};
	auto original2 = eteq::make_functor(egen::MATMUL, expect_orig2chain);

	opt::GraphInfo graph({original, original2});

	types::PairsT<teq::iTensor*,teq::TensptrsT> chain_roots;
	hone::flatten_matmul_hierarchy(chain_roots, graph);
	teq::TensMapT<teq::TensptrsT> chains(
		chain_roots.begin(), chain_roots.end());

	EXPECT_EQ(3, chains.size());

	ASSERT_HAS(chains, original.get());
	ASSERT_HAS(chains, original2.get());
	ASSERT_HAS(chains, inner.get());

	teq::TensptrsT origvars = chains.at(original.get());
	teq::TensptrsT orig2vars = chains.at(original2.get());
	teq::TensptrsT innervars = chains.at(inner.get());

	EXPECT_VECEQ(expect_origchain, origvars);
	EXPECT_VECEQ(expect_orig2chain, orig2vars);
	EXPECT_VECEQ(expect_innerchain, innervars);
}


// Don't care about sparseness or constants
TEST(MATCHAIN, DenseMatmuls)
{
	teq::ShapesT shapes = {
		teq::Shape({15, 15}), teq::Shape({450, 15}),
		teq::Shape({450, 450}), teq::Shape({30, 450}),
		teq::Shape({30, 30}), teq::Shape({900, 30}),
		teq::Shape({900, 900}), teq::Shape({900, 900}),
		teq::Shape({900, 900})
	};
	teq::TensptrT original = eteq::make_variable_scalar(9.f, shapes.front());
	for (size_t i = 1, n = shapes.size(); i < n; ++i)
	{
		original = eteq::make_functor(egen::MATMUL, {
			original, eteq::make_variable_scalar(9.f, shapes[i])
		});
	}

	opt::GraphInfo graph({original});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	auto replacement = roots.front();

	EXPECT_GRAPHEQ(
	"(MATMUL<FLOAT>[900\\15\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[900\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[900\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(MATMUL<FLOAT>[900\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(MATMUL<FLOAT>[30\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[15\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(MATMUL<FLOAT>[30\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______`--(MATMUL<FLOAT>[30\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______|___`--(MATMUL<FLOAT>[450\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______|___|___`--(variable:9<FLOAT>[450\\15\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______|___|___`--(variable:9<FLOAT>[450\\450\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______|___`--(variable:9<FLOAT>[30\\450\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|_______`--(variable:9<FLOAT>[30\\30\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(variable:9<FLOAT>[900\\30\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(variable:9<FLOAT>[900\\900\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(variable:9<FLOAT>[900\\900\\1\\1\\1\\1\\1\\1])\n"
	"_`--(variable:9<FLOAT>[900\\900\\1\\1\\1\\1\\1\\1])", replacement);
}


TEST(MATCHAIN, SparseMatmuls)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	teq::ShapesT shapes = {
		teq::Shape({150, 150}), teq::Shape({45000, 150}),
		teq::Shape({45000, 45000}), teq::Shape({300, 45000}),
		teq::Shape({300, 300}), teq::Shape({90000, 300}),
		teq::Shape({90000, 90000}), teq::Shape({90000, 90000}),
		teq::Shape({90000, 90000})
	};
	teq::TensptrT original = make_sparse(shapes.front(), true);
	for (size_t i = 1, n = shapes.size(); i < n; ++i)
	{
		original = eteq::make_functor(egen::MATMUL, {
			original, make_sparse(shapes[i], i % 2)
		});
	}

	opt::GraphInfo graph({original});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	auto replacement = roots.front();

	EXPECT_GRAPHEQ(
	"(MATMUL<FLOAT>[90000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[300\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[300\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(MATMUL<FLOAT>[45000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(MATMUL<FLOAT>[45000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[150\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[45000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(variable:9<FLOAT>[45000\\45000\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(variable:9<FLOAT>[300\\45000\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(variable:9<FLOAT>[300\\300\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(MATMUL<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(MATMUL<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___|___`--(variable:9<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___|___`--(variable:9<FLOAT>[90000\\90000\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(variable:9<FLOAT>[90000\\90000\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(variable:9<FLOAT>[90000\\90000\\1\\1\\1\\1\\1\\1])\n",
	replacement);
}


TEST(MATCHAIN, RealisticSparseMatmuls)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	teq::ShapesT shapes = {
		teq::Shape({150, 150}), teq::Shape({300, 150}),
		teq::Shape({300, 300}), teq::Shape({90000, 300}),
		teq::Shape({90000, 90000}), teq::Shape({90000, 90000}),
	};
	std::vector<float> densities = {
		0.006, 1.0, 0.003, 0.003, 0.00001, 0.00001
	};
	teq::TensptrT original = make_randsparse<float>(
		shapes.front(), densities.front(), "9");
	for (size_t i = 1, n = shapes.size(); i < n; ++i)
	{
		original = eteq::make_functor(egen::MATMUL, {
			original, make_randsparse<float>(
				shapes[i], densities[i], "9")
		});
	}
	opt::GraphInfo graph({original});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	auto replacement = roots.front();

	EXPECT_GRAPHEQ(
	"(MATMUL<FLOAT>[90000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[300\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(variable:9<FLOAT>[150\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[300\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|_______`--(variable:9<FLOAT>[300\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|_______`--(variable:9<FLOAT>[300\\300\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(MATMUL<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(variable:9<FLOAT>[90000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(variable:9<FLOAT>[90000\\90000\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(variable:9<FLOAT>[90000\\90000\\1\\1\\1\\1\\1\\1])\n",
	replacement);
}


TEST(MATCHAIN, SparseMatmuls2)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	size_t inject_begin = 3;
	size_t inject_end = 7;
	teq::ShapesT outershapes = {
		teq::Shape({150, 150}), teq::Shape({4500, 150}),
		teq::Shape({4500, 4500}), teq::Shape({9000, 4500}),
		teq::Shape({9000, 9000}), teq::Shape({9000, 9000})
	};
	std::vector<double> densities = {
		0.00497778, 0.00499852, 0.00490005, 0.03332, 0.00490001, 0.00490001
	};
	teq::TensptrsT vars;
	size_t n = outershapes.size();
	vars.reserve(n);
	for (size_t i = 0; i < n; ++i)
	{
		vars.push_back(make_randsparse<float>(
			outershapes[i], densities[i], "9"));
	}

	auto original = vars.back();
	for (size_t i = 1, n = vars.size(); i < n; ++i)
	{
		original = eteq::make_functor(egen::MATMUL, { vars[n - 1 - i], original });
	}

	opt::GraphInfo graph({original});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	auto replacement = roots.front();

	EXPECT_GRAPHEQ(
	"_(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(MATMUL<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(MATMUL<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[150\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(variable:9<FLOAT>[4500\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(variable:9<FLOAT>[9000\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n"
	"_`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n",
	replacement);
}


TEST(MATCHAIN, InnerMatmul)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	teq::ShapesT outershapes = {
		teq::Shape({300, 4500}), teq::Shape({300, 300}),
		teq::Shape({9000, 300}), teq::Shape({9000, 9000}),
	};
	teq::TensptrsT vars;
	vars.reserve(outershapes.size());
	for (auto shape : outershapes)
	{
		vars.push_back(make_sparse(shape, true));
	}

	auto original = vars.back();
	for (auto it = vars.rbegin() + 1, et = vars.rend(); it != et; ++it)
	{
		original = eteq::make_functor(egen::MATMUL, { *it, original });
	}

	opt::GraphInfo graph({original});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(1, roots.size());
	auto replacement = roots.front();

	EXPECT_GRAPHEQ(
	"(MATMUL<FLOAT>[9000\\4500\\1\\1\\1\\1\\1\\1])\n" // inner
	"_`--(variable:9<FLOAT>[300\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(variable:9<FLOAT>[300\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____|___`--(variable:9<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_____`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n",
	 replacement);
}


TEST(MATCHAIN, SharedNodeMatmul)
{
	size_t seed = 800;
	static_cast<global::iRandGenerator*>(test_generator.get())->seed(seed);

	size_t inject_begin = 3;
	size_t inject_end = 7;
	teq::ShapesT outershapes = {
		teq::Shape({150, 150}), teq::Shape({4500, 150}),
		teq::Shape({4500, 4500}), teq::Shape({300, 4500}),
		teq::Shape({300, 300}), teq::Shape({9000, 300}),
		teq::Shape({9000, 9000}), teq::Shape({9000, 9000}),
		teq::Shape({9000, 9000})
	};
	teq::TensptrsT vars;
	vars.reserve(outershapes.size());
	for (auto shape : outershapes)
	{
		vars.push_back(make_sparse(shape, true));
	}

	auto vit = vars.begin();
	teq::TensptrsT expect_innerchain(vit + inject_begin, vit + inject_end);
	auto inner = expect_innerchain.back();
	for (auto it = expect_innerchain.rbegin() + 1, et = expect_innerchain.rend();
		it != et; ++it)
	{
		inner = eteq::make_functor(egen::MATMUL, { *it, inner });
	}

	teq::TensptrsT expect_origchain(vit, vit + inject_begin);
	expect_origchain.push_back(inner);
	expect_origchain.insert(expect_origchain.end(),
		vit + inject_end, vars.end());
	auto original = expect_origchain.front();
	for (auto it = expect_origchain.begin() + 1, et = expect_origchain.end();
		it != et; ++it)
	{
		original = eteq::make_functor(egen::MATMUL, { original, *it });
	}

	// expect original shape [9000,4500]
	auto original2 = eteq::make_functor(egen::MATMUL, {
		inner, make_sparse(teq::Shape({100, 9000}), true) });

	opt::GraphInfo graph({original, original2});

	hone::matrix_chain(graph);
	auto roots = graph.get_roots();
	ASSERT_EQ(2, roots.size());
	auto replacement = roots.front();
	auto replacement2 = roots.back();

	EXPECT_GRAPHEQ(
	"_(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[9000\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(MATMUL<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(MATMUL<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[150\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___|___`--(variable:9<FLOAT>[4500\\150\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___|___`--(variable:9<FLOAT>[4500\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___`--(MATMUL<FLOAT>[9000\\4500\\1\\1\\1\\1\\1\\1])\n" // inner
	"_|___|_______`--(variable:9<FLOAT>[300\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_|___|_______`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___________`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___________|___`--(variable:9<FLOAT>[300\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___________|___`--(variable:9<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|___|___________`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n"
	"_`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n",
	replacement);
	EXPECT_GRAPHEQ(
	"_(MATMUL<FLOAT>[100\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_`--(MATMUL<FLOAT>[9000\\4500\\1\\1\\1\\1\\1\\1])\n" // inner
	"_|___`--(variable:9<FLOAT>[300\\4500\\1\\1\\1\\1\\1\\1])\n"
	"_|___`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|_______`--(MATMUL<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|_______|___`--(variable:9<FLOAT>[300\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|_______|___`--(variable:9<FLOAT>[9000\\300\\1\\1\\1\\1\\1\\1])\n"
	"_|_______`--(variable:9<FLOAT>[9000\\9000\\1\\1\\1\\1\\1\\1])\n"
	"_`--(variable:9<FLOAT>[100\\9000\\1\\1\\1\\1\\1\\1])\n",
	replacement2);
}


#endif // DISABLE_HONE_MATCHAIN_TES
