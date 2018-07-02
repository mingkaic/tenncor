#ifndef DISABLE_SLIP_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"
#include "clay/error.hpp"

#include "slip/registry.hpp"
#include "slip/rand.hpp"
#include "slip/error.hpp"


#ifndef DISABLE_REGISTRY_TEST


using namespace testutil;


class REGISTRY : public fuzz_test {};


using SCALAR = std::function<double(double)>;


using AGGS = std::function<double(std::vector<double>)>;


template <typename T>
using SCALARS = std::function<T(T, T)>;


using TWODV = std::vector<std::vector<int64_t> >;


static const double ERR_THRESH = 0.08; // 8% error


static void unaryElemTest (fuzz_test* fuzzer, slip::OPCODE opcode,
	SCALAR expect, std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	size_t nbytes = argument.size() * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::memcpy(data.get(), &argument[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"unary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), mold::Range(0, 0))
	}));

	double* outptr = (double*) output.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument[i]), outptr[i]);
	}
}


static void unaryAggTest (fuzz_test* fuzzer, slip::OPCODE opcode,
	AGGS agg, std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	size_t nbytes = argument.size() * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::memcpy(data.get(), &argument[0], nbytes);
	clay::State in(data, shape, clay::DOUBLE);
	mold::StateRange inr(in, mold::Range(0, -1));
	clay::Shape wun = std::vector<size_t>{1};

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"unary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);

	clay::TensorPtrT tens = op->make_data({inr});
	ASSERT_SHAPEQ(wun, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(sizeof(double));
	clay::State out(output, wun, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {inr}));
	EXPECT_EQ(agg(argument), *((double*) output.get()));
}


static void binaryAggTest (fuzz_test* fuzzer, slip::OPCODE opcode,
	std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	size_t nbytes = argument.size() * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::memcpy(data.get(), &argument[0], nbytes);
	clay::State in(data, shape, clay::DOUBLE);
	clay::Shape wun = std::vector<size_t>{1};

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"unary " << slip::opnames.at(opcode) << " not found";
	auto bad_op = slip::get_op(opcode);
	auto dim_op = slip::get_op(opcode);
	uint64_t badrank = shape.rank();
	uint64_t dim = (uint64_t) fuzzer->get_int(1, "dim", {0, badrank-1})[0];
	std::shared_ptr<char> bad_data = clay::make_char(sizeof(uint64_t));
	std::shared_ptr<char> dim_data = clay::make_char(sizeof(uint64_t));
	std::memcpy(bad_data.get(), &badrank, sizeof(uint64_t));
	std::memcpy(dim_data.get(), &dim, sizeof(uint64_t));
	EXPECT_THROW(bad_op->make_data({
			mold::StateRange(in, mold::Range(0, 0)),
			mold::StateRange(clay::State(bad_data, wun, clay::UINT64), mold::Range(0, 0))
		}),
		slip::InvalidDimensionError);

	std::vector<size_t> slist = shape.as_list();
	slist.erase(slist.begin() + dim);
	clay::Shape dshape = slist;
	clay::TensorPtrT tens = dim_op->make_data({
		mold::StateRange(in, mold::Range(0, 0)),
		mold::StateRange(clay::State(dim_data, wun, clay::UINT64), mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(dshape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> dim_output = clay::make_char(dshape.n_elems() * sizeof(double));
	clay::State dim_out(dim_output, dshape, clay::DOUBLE);
	ASSERT_TRUE(dim_op->write_data(dim_out, {
		mold::StateRange(in, mold::Range(0, 0)),
		mold::StateRange(clay::State(dim_data, wun, clay::UINT64), mold::Range(0, 0))
	}));
}


static void binaryElemTest (fuzz_test* fuzzer, slip::OPCODE opcode,
	SCALARS<double> expect, std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument0 = fuzzer->get_double(n, "argument0", limits);
	std::vector<double> argument1 = fuzzer->get_double(n, "argument1", limits);
	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(nbytes);
	std::memcpy(data0.get(), &argument0[0], nbytes);
	std::memcpy(data1.get(), &argument1[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"binary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, shape, clay::DOUBLE);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	double* outptr = (double*) output.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument0[i], argument1[i]), outptr[i]);
	}
}


static void binaryElemTestInt (fuzz_test* fuzzer, slip::OPCODE opcode,
	SCALARS<uint64_t> expect, std::pair<uint64_t,uint64_t> limits = {0, 2})
{
	clay::Shape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<size_t> temp0 = fuzzer->get_int(n, "argument0", limits);
	std::vector<size_t> temp1 = fuzzer->get_int(n, "argument1", limits);
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<uint64_t> argument1(temp1.begin(), temp1.end());
	size_t nbytes = n * sizeof(uint64_t);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(nbytes);
	std::memcpy(data0.get(), &argument0[0], nbytes);
	std::memcpy(data1.get(), &argument1[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"binary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::UINT64);
	clay::State in1(data1, shape, clay::UINT64);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::UINT64, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::UINT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	uint64_t* outptr = (uint64_t*) output.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument0[i], argument1[i]), outptr[i]);
	}
}


static inline TWODV create2D (std::vector<int64_t> juanD, size_t C, size_t R)
{
	TWODV res;
 	for (size_t y = 0; y < R; y++)
	{
		res.push_back(std::vector<int64_t>(C, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			size_t juan_coord = x + y * C;
			res[y][x] = juanD[juan_coord];
		}
	}
	return res;
}


static inline bool freivald (fuzz_test* fuzzer, TWODV a, TWODV b, TWODV c)
{
	assert(!b.empty());
	size_t rlen = b[0].size();
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	size_t m = 10;
	for (size_t i = 0; i < m; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<size_t> r = fuzzer->get_int(rlen, ioutil::Stream() << "freivald_vec" << i, {0, 1});

		// p = a @ (b @ r) - c @ r
		std::vector<int64_t> br;
		for (size_t y = 0, n = b.size(); y < n; y++)
		{
			int64_t bri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<int64_t> cr;
		for (size_t y = 0, n = c.size(); y < n; y++)
		{
			int64_t cri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<int64_t> p;
		size_t n = a.size();
		for (size_t y = 0; y < n; y++)
		{
			int64_t ari = 0;
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
		if (!std::all_of(p.begin(), p.end(), [](int64_t d) { return d == 0; }))
		{
			return false;
		}
	}
	return true;
}


TEST_F(REGISTRY, Unsupported_B000)
{
	EXPECT_FALSE(slip::has_op(slip::_SENTINEL));
	EXPECT_THROW(slip::get_op(slip::_SENTINEL), slip::UnsupportedOpcodeError);
}


TEST_F(REGISTRY, Cast_B001)
{
	slip::OPCODE opcode = slip::CAST;
	clay::Shape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument1", {-123, 231});
	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::memcpy(data.get(), &argument[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"binary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data, shape, clay::INT64);
	clay::State in1(data, shape, clay::DOUBLE);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::INT64, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::INT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	int64_t* outptr = (int64_t*) output.get();
	for (size_t i = 0; i < n; ++i)
	{
		int64_t expect = argument[i];
		EXPECT_EQ(expect, outptr[i]);
	}
}


TEST_F(REGISTRY, Abs_B002)
{
	unaryElemTest(this, slip::ABS,
	[](double var) { return std::abs(var); });
}


TEST_F(REGISTRY, Neg_B003)
{
	unaryElemTest(this, slip::NEG,
	[](double var) { return -var; });
}


TEST_F(REGISTRY, Not_B004)
{
	unaryElemTest(this, slip::NOT,
	[](double var) { return !var; });
}


TEST_F(REGISTRY, Sin_B005)
{
	unaryElemTest(this, slip::SIN,
	[](double var) { return std::sin(var); });
}


TEST_F(REGISTRY, Cos_B006)
{
	unaryElemTest(this, slip::COS,
	[](double var) { return std::cos(var); });
}


TEST_F(REGISTRY, Tan_B007)
{
	unaryElemTest(this, slip::TAN,
	[](double var) { return std::tan(var); });
}


TEST_F(REGISTRY, Exp_B008)
{
	unaryElemTest(this, slip::EXP,
	[](double var) { return std::exp(var); });
}


TEST_F(REGISTRY, Log_B009)
{
	unaryElemTest(this, slip::LOG,
	[](double var) { return std::log(var); }, {0.5, 7});
}


TEST_F(REGISTRY, Sqrt_B010)
{
	unaryElemTest(this, slip::SQRT,
	[](double var) { return std::sqrt(var); }, {0, 7});
}


TEST_F(REGISTRY, Round_B011)
{
	unaryElemTest(this, slip::ROUND,
	[](double var) { return std::round(var); });
}


TEST_F(REGISTRY, UArgmax_B012)
{
	unaryAggTest(this, slip::UARGMAX,
	[](std::vector<double> vec) -> double
	{
		auto it = std::max_element(vec.begin(), vec.end());
		return (double) std::distance(vec.begin(), it);
	});
}


TEST_F(REGISTRY, UMax_B013)
{
	unaryAggTest(this, slip::URMAX,
	[](std::vector<double> vec) -> double
	{
		return *std::max_element(vec.begin(), vec.end());
	});
}


TEST_F(REGISTRY, USum_B014)
{
	unaryAggTest(this, slip::URSUM,
	[](std::vector<double> vec) -> double
	{
		return std::accumulate(vec.begin(), vec.end(), (double) 0);
	});
}


TEST_F(REGISTRY, Argmax_B012)
{
	binaryAggTest(this, slip::ARGMAX);
}


TEST_F(REGISTRY, Max_B013)
{
	binaryAggTest(this, slip::RMAX);
}


TEST_F(REGISTRY, Sum_B014)
{
	binaryAggTest(this, slip::RSUM);
}


TEST_F(REGISTRY, Tranpose_B015)
{
	slip::OPCODE opcode = slip::TRANSPOSE;
	std::vector<size_t> clist = random_def_shape(this);
	size_t rank = clist.size();
	std::vector<uint64_t> perm(rank);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(),
		slip::get_generator());

	clay::Shape shape = clist;
	std::vector<size_t> permlist(rank);
	for (size_t i = 0; i < rank; ++i)
	{
		permlist[i] = clist[perm[i]];
	}
	clay::Shape permshape(permlist);
	std::reverse(clist.begin(), clist.end());
	clay::Shape outshape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	size_t nbytes0 = n * sizeof(double);
	size_t nbytes1 = rank * sizeof(uint64_t);
	std::shared_ptr<char> data0 = clay::make_char(nbytes0);
	std::shared_ptr<char> data1 = clay::make_char(nbytes1);
	std::memcpy(data0.get(), &argument[0], nbytes0);
	std::memcpy(data1.get(), &perm[0], nbytes1);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"function " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	auto op1 = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, clay::Shape({rank}), clay::UINT64);

	clay::TensorPtrT tens = op->make_data({mold::StateRange(in0, mold::Range(0, 0))});
	ASSERT_SHAPEQ(outshape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());
	clay::TensorPtrT tens1 = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(permshape, tens1->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens1->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes0);
	clay::State out(output, outshape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0))
	}));

	std::shared_ptr<char> outputperm = clay::make_char(nbytes0);
	clay::State outperm(outputperm, permshape, clay::DOUBLE);
	ASSERT_TRUE(op1->write_data(outperm, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	double* data = (double*) output.get();
	double* dataperm = (double*) outputperm.get();

	std::vector<size_t> coord;
	std::vector<size_t> tmp_coord;
	for (size_t i = 0; i < n; ++i)
	{
		tmp_coord = coord = clay::coordinate(permshape, i);
		for (size_t j = 0; j < perm.size(); ++j)
		{
			coord[perm[j]] = tmp_coord[j];
		}
		size_t permidx = clay::index(shape, coord);
		EXPECT_EQ(argument[permidx], dataperm[i]);

		coord = clay::coordinate(outshape, i);
		std::reverse(coord.begin(), coord.end());
		size_t idx = clay::index(shape, coord);
		EXPECT_EQ(argument[idx], data[i]);
	}
}


TEST_F(REGISTRY, Flip_B016)
{
	slip::OPCODE opcode = slip::FLIP;
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size() - 1})[0];
	uint64_t badarg = clist.size();
	clay::Shape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});

	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(sizeof(uint64_t));
	std::memcpy(data0.get(), &argument[0], nbytes);
	std::memcpy(data1.get(), &argidx, sizeof(uint64_t));

	std::shared_ptr<char> baddata = clay::make_char(sizeof(uint64_t));
	std::memcpy(baddata.get(), &badarg, sizeof(uint64_t));

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"function " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, clay::Shape({1}), clay::UINT64);

	auto bad_op = slip::get_op(opcode);
	EXPECT_THROW(bad_op->make_data({
			mold::StateRange(in0, mold::Range(0, 0)),
			mold::StateRange(clay::State(
				baddata,
				clay::Shape({1}),
				clay::UINT64), mold::Range(0, 0))
		}),
		slip::InvalidDimensionError);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	double* data = (double*) output.get();
	std::vector<size_t> coord;
	for (size_t i = 0; i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord[argidx] = clist[argidx] - coord[argidx] - 1;
		size_t idx = index(shape, coord);
		EXPECT_EQ(argument[idx], data[i]);
	}
}


TEST_F(REGISTRY, Expand_B017)
{
	slip::OPCODE opcode = slip::EXPAND;
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size()})[0];
	uint64_t badarg = clist.size() + 1;
	uint64_t mult = get_int(1, "mult", {1, 6})[0];
	clay::Shape shape = clist;
	clist.insert(clist.begin() + argidx, mult);
	clay::Shape outshape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});

	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(sizeof(uint64_t));
	std::shared_ptr<char> data2 = clay::make_char(sizeof(uint64_t));
	std::memcpy(data0.get(), &argument[0], nbytes);
	std::memcpy(data1.get(), &mult, sizeof(uint64_t));
	std::memcpy(data2.get(), &argidx, sizeof(uint64_t));

	std::shared_ptr<char> baddata = clay::make_char(sizeof(uint64_t));
	std::memcpy(baddata.get(), &badarg, sizeof(uint64_t));

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"function " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, clay::Shape({1}), clay::UINT64);
	clay::State in2(data2, clay::Shape({1}), clay::UINT64);

	auto bad_op = slip::get_op(opcode);
	EXPECT_THROW(bad_op->make_data({
			mold::StateRange(in0, mold::Range(0, 0)),
			mold::StateRange(in1, mold::Range(0, 0)),
			mold::StateRange(clay::State(
				baddata,
				clay::Shape({1}),
				clay::UINT64), mold::Range(0, 0))
		}),
		slip::InvalidDimensionError);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0)),
		mold::StateRange(in2, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(outshape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	size_t noutbytes = outshape.n_elems() * sizeof(double);
	std::shared_ptr<char> output = clay::make_char(noutbytes);
	clay::State out(output, outshape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0)),
		mold::StateRange(in2, mold::Range(0, 0))
	}));

	double* data = (double*) output.get();
	std::vector<size_t> coord;
	for (size_t i = 0; i < n; ++i)
	{
		coord = coordinate(shape, i);
		coord.insert(coord.begin() + argidx, 0);
		for (size_t j = 0; j < mult; ++j)
		{
			coord[argidx] = j;
			size_t idx = index(outshape, coord);
			EXPECT_EQ(argument[i], data[idx]);
		}
	}
}


TEST_F(REGISTRY, NElems_B018)
{
	slip::OPCODE opcode = slip::N_ELEMS;
	clay::Shape shape = random_def_shape(this);
	clay::Shape outshape({1});
	size_t n = shape.n_elems();
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
	size_t nbytes = n * clay::type_size(dtype);
	std::string argument = get_string(nbytes, "argument");

	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::memcpy(data.get(), argument.c_str(), nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"function " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(clay::State(data, shape, dtype), mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(outshape, tens->get_shape());
	ASSERT_EQ(clay::UINT64, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(sizeof(uint64_t));
	clay::State out(output, outshape, clay::UINT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(clay::State(data, shape, dtype), mold::Range(0, 0))
	}));

	uint64_t* outptr = (uint64_t*) output.get();
	EXPECT_EQ(n, *outptr);
}


TEST_F(REGISTRY, NDims_B019)
{
	slip::OPCODE opcode = slip::N_DIMS;
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size() - 1})[0];
	uint64_t badarg = clist.size() + 1;
	clay::Shape shape = clist;
	clay::Shape outshape({1});
	size_t n = shape.n_elems();
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
	size_t nbytes = n * clay::type_size(dtype);
	std::string argument = get_string(nbytes, "argument");

	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::shared_ptr<char> dptr = clay::make_char(sizeof(uint64_t));
	std::memcpy(data.get(), argument.c_str(), nbytes);
	std::memcpy(dptr.get(), &argidx, sizeof(uint64_t));

	std::shared_ptr<char> baddata = clay::make_char(sizeof(uint64_t));
	std::memcpy(baddata.get(), &badarg, sizeof(uint64_t));

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"function " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);

	clay::State in(data, shape, dtype);

	auto bad_op = slip::get_op(opcode);
	EXPECT_THROW(bad_op->make_data({
			mold::StateRange(in, mold::Range(0, 0)),
			mold::StateRange(clay::State(baddata, outshape, clay::UINT64), mold::Range(0, 0))
		}),
		slip::InvalidDimensionError);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in, mold::Range(0, 0)),
		mold::StateRange(clay::State(dptr, outshape, clay::UINT64), mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(outshape, tens->get_shape());
	ASSERT_EQ(clay::UINT64, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(sizeof(uint64_t));
	clay::State out(output, outshape, clay::UINT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in, mold::Range(0, 0)),
		mold::StateRange(clay::State(dptr, outshape, clay::UINT64), mold::Range(0, 0))
	}));

	uint64_t* outptr = (uint64_t*) output.get();
	EXPECT_EQ(clist[argidx], *outptr);
}


TEST_F(REGISTRY, Pow_B030)
{
	binaryElemTest(this, slip::POW,
	[](double a, double b) { return std::pow(a, b); }, {0, 7});
}


TEST_F(REGISTRY, Add_B031)
{
	binaryElemTest(this, slip::ADD,
	[](double a, double b) { return a + b; });
}


TEST_F(REGISTRY, Sub_B032)
{
	binaryElemTest(this, slip::SUB,
	[](double a, double b) { return a - b; });
}


TEST_F(REGISTRY, Mul_B033)
{
	binaryElemTest(this, slip::MUL,
	[](double a, double b) { return a * b; });
}


TEST_F(REGISTRY, Div_B034)
{
	binaryElemTest(this, slip::DIV,
	[](double a, double b) { return a / b; });
}


TEST_F(REGISTRY, Eq_B035)
{
	binaryElemTestInt(this, slip::EQ,
	[](uint64_t a, uint64_t b) -> uint64_t { return a == b; });
}


TEST_F(REGISTRY, Neq_B036)
{
	binaryElemTestInt(this, slip::NE,
	[](uint64_t a, uint64_t b) -> uint64_t { return a != b; });
}


TEST_F(REGISTRY, Lt_B037)
{
	binaryElemTestInt(this, slip::LT,
	[](uint64_t a, uint64_t b) -> uint64_t { return a < b; });
}


TEST_F(REGISTRY, Gt_B038)
{
	binaryElemTestInt(this, slip::GT,
	[](uint64_t a, uint64_t b) -> uint64_t { return a > b; });
}


TEST_F(REGISTRY, Uniform_B039)
{
	slip::OPCODE opcode = slip::UNIF;
	clay::Shape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-2, 1});
	std::vector<double> argument1 = get_double(n, "argument1", {2, 5});
	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(nbytes);
	std::memcpy(data0.get(), &argument0[0], nbytes);
	std::memcpy(data1.get(), &argument1[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"sample " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, shape, clay::DOUBLE);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	double* outptr = (double*) output.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_LT(argument0[i], outptr[i]);
		EXPECT_GT(argument1[i], outptr[i]);
	}
}


TEST_F(REGISTRY, Binom_B040)
{
	slip::OPCODE opcode = slip::BINO;
	clay::Shape shape = random_def_shape(this, {1, 6}, {500, 7983});
	size_t n = shape.n_elems();
	auto temp0 = get_int(n, "argument0", {2, 19});
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<double> argument1 = get_double(n, "argument1", {0, 1});
	size_t nbytes = n * sizeof(uint64_t);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(n * sizeof(double));
	std::memcpy(data0.get(), &argument0[0], nbytes);
	std::memcpy(data1.get(), &argument1[0], n * sizeof(double));

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"sample " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::UINT64);
	clay::State in1(data1, shape, clay::DOUBLE);

	auto badop_double = slip::get_op(opcode);
	auto badop_float = slip::get_op(opcode);
	auto bad_arg = slip::get_op(opcode);
	EXPECT_THROW(badop_double->make_data({
			mold::StateRange(clay::State(data0, shape, clay::DOUBLE), mold::Range(0, 0)),
			mold::StateRange(in1, mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	EXPECT_THROW(badop_float->make_data({
			mold::StateRange(clay::State(data0, shape, clay::FLOAT), mold::Range(0, 0)),
			mold::StateRange(in1, mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	clay::DTYPE notdoub = (clay::DTYPE) get_int(1, "notdoub", {1, clay::DTYPE::_SENTINEL - 1})[0];
	if (notdoub == clay::DOUBLE)
	{
		notdoub = clay::FLOAT;
	}
	EXPECT_THROW(bad_arg->make_data({
			mold::StateRange(in0, mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, notdoub), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::UINT64, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::UINT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	uint64_t* outptr = (uint64_t*) output.get();
	// approximate to normal distribution
	std::vector<double> stdev_count(3, 0);
	for (size_t i = 0; i < n; ++i)
	{
		double mean = argument0[i] * argument1[i];
		double stdev = mean * (1 - argument1[i]);
		size_t index = std::abs(mean - outptr[i]) / stdev;
		if (index < stdev_count.size())
		{
			stdev_count[index]++;
		}
	}
	// check the first 3 stdev
	double expect68 = stdev_count[0] / n; // expect ~68%
	double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	double err1 = std::abs(0.68 - expect68);
	double err2 = std::abs(0.95 - expect95);
	double err3 = std::abs(0.997 - expect99);

	// allow larger error threshold to account for small n
	EXPECT_GT(ERR_THRESH * 2, err1);
	EXPECT_GT(ERR_THRESH * 2, err2);
	EXPECT_GT(ERR_THRESH * 2, err3);
}


TEST_F(REGISTRY, Norm_B041)
{
	slip::OPCODE opcode = slip::NORM;
	clay::Shape shape = random_def_shape(this, {1, 6}, {300, 7983});
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-21, 154});
	std::vector<double> argument1 = get_double(n, "argument1", {1, 123});
	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data0 = clay::make_char(nbytes);
	std::shared_ptr<char> data1 = clay::make_char(nbytes);
	std::memcpy(data0.get(), &argument0[0], nbytes);
	std::memcpy(data1.get(), &argument1[0], nbytes);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"sample " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape, clay::DOUBLE);
	clay::State in1(data1, shape, clay::DOUBLE);

	auto badop = slip::get_op(opcode);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::INT8), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::INT8), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::UINT8), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::UINT8), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::INT16), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::INT16), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::UINT16), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::UINT16), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::INT32), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::INT32), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::UINT32), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::UINT32), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::INT64), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::INT64), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);
	ASSERT_THROW(badop->make_data({
			mold::StateRange(clay::State(data0, shape, clay::UINT64), mold::Range(0, 0)),
			mold::StateRange(clay::State(data1, shape, clay::UINT64), mold::Range(0, 0))
		}),
		clay::UnsupportedTypeError);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(shape, tens->get_shape());
	ASSERT_EQ(clay::DOUBLE, tens->get_type());

	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, shape, clay::DOUBLE);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	// double* outptr = (double*) output.get();
	// std::vector<double> stdev_count(3, 0);
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	size_t index = std::abs(argument0[i] - outptr[i]) / argument1[i];
	// 	if (index < stdev_count.size())
	// 	{
	// 		stdev_count[index]++;
	// 	}
	// }
	// // check the first 3 stdev
	// double expect68 = stdev_count[0] / n; // expect ~68%
	// double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	// double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	// double err1 = std::abs(0.68 - expect68);
	// double err2 = std::abs(0.95 - expect95);
	// double err3 = std::abs(0.997 - expect99);

	// EXPECT_GT(ERR_THRESH, err1);
	// EXPECT_GT(ERR_THRESH, err2);
	// EXPECT_GT(ERR_THRESH, err3);
}


TEST_F(REGISTRY, Matmul_B042)
{
	slip::OPCODE opcode = slip::MATMUL;
	std::vector<size_t> clist = random_def_shape(this); // <m, n, ...>
	size_t k = get_int(1, "k", {1, 8})[0];
	size_t m = clist[0];
	size_t n = clist[1];
	clay::Shape shape0 = clist;
	clist[1] = m;
	clist[0] = k;
	clay::Shape shape1 = clist; // <k, m, ...>
	clist[1] = n;
	clay::Shape outshape = clist; // <k, n, ...>
	size_t n0 = shape0.n_elems();
	size_t n1 = shape1.n_elems();
	size_t nout = outshape.n_elems();
	auto temp0 = get_int(n0, "argument0", {0, 19});
	auto temp1 = get_int(n1, "argument1", {0, 19});
	std::vector<int64_t> argument0(temp0.begin(), temp0.end());
	std::vector<int64_t> argument1(temp1.begin(), temp1.end());
	size_t nbytes0 = n0 * sizeof(int64_t);
	size_t nbytes1 = n1 * sizeof(int64_t);
	std::shared_ptr<char> data0 = clay::make_char(nbytes0);
	std::shared_ptr<char> data1 = clay::make_char(nbytes1);
	std::memcpy(data0.get(), &argument0[0], nbytes0);
	std::memcpy(data1.get(), &argument1[0], nbytes1);

	ASSERT_TRUE(slip::has_op(opcode)) <<
		"binary " << slip::opnames.at(opcode) << " not found";
	auto op = slip::get_op(opcode);
	clay::State in0(data0, shape0, clay::INT64);
	clay::State in1(data1, shape1, clay::INT64);

	clay::TensorPtrT tens = op->make_data({
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	});
	ASSERT_SHAPEQ(outshape, tens->get_shape());
	ASSERT_EQ(clay::INT64, tens->get_type());

	size_t nbytes = nout * sizeof(int64_t);
	std::shared_ptr<char> output = clay::make_char(nbytes);
	clay::State out(output, outshape, clay::INT64);
	ASSERT_TRUE(op->write_data(out, {
		mold::StateRange(in0, mold::Range(0, 0)),
		mold::StateRange(in1, mold::Range(0, 0))
	}));

	int64_t* outptr = (int64_t*) output.get();

	size_t nchunks = std::accumulate(clist.begin() + 2, clist.end(), 1, std::multiplies<size_t>());
	size_t nchunk0 = m * n;
	size_t nchunk1 = k * m;
	size_t nchunkr = k * n;
	auto it0 = argument0.begin();
	auto it1 = argument1.begin();
	for (size_t i = 0; i < nchunks; ++i)
	{
		std::vector<int64_t> chunka(it0 + i * nchunk0, it0 + (i + 1) * nchunk0);
		std::vector<int64_t> chunkb(it1 + i * nchunk1, it1 + (i + 1) * nchunk1);
		std::vector<int64_t> chunkr(outptr + i * nchunkr, outptr + (i + 1) * nchunkr);
		EXPECT_TRUE(freivald(this, create2D(chunka, m, n), create2D(chunkb, k, m), create2D(chunkr, k, n))) <<
			"matrix multiplication failed at level " << i;
	}
}


#endif /* DISABLE_REGISTRY_TEST */


#endif /* DISABLE_SLIP_MODULE_TESTS */
