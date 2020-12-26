
#ifndef DISABLE_TENNCOR_NN_TEST

#include "dbg/print/teq.hpp"

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/tenncor.hpp"


TEST(NN, Dropout)
{
	auto old_generator = global::get_generator();
	global::set_generator(std::make_shared<global::Randomizer>());
	global::seed(0);

	teq::Shape shape({2, 5});
	auto x = eteq::make_constant_scalar<float>(1, shape);
	auto f = tenncor().nn.dropout<float>(x, 0.1f);

	EXPECT_GRAPHEQ(
		"(MUL<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:1<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(LT<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(RAND_UNIF<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(variable:0<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(variable:1<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(EXTEND<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(SUB<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(EXTEND<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___`--(constant:1<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(variable:drop_rate<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(DIV<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(REDUCE_SUM<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(LT<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(RAND_UNIF<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______|___`--(variable:0<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______|___`--(variable:1<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(EXTEND<FLOAT>[2\\5\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(SUB<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______________`--(EXTEND<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______________|___`--(constant:1<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______________`--(variable:drop_rate<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(constant:10<FLOAT>[1\\1\\1\\1\\1\\1\\1\\1])",
		f);

	ASSERT_ARREQ(shape, f->shape());
	auto data = f.calc<float>();

	float values = 0;
	size_t not_zeros = 0;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		if (data[i] > 0)
		{
			++not_zeros;
		}
		values += data[i];
	}
	EXPECT_LE(not_zeros, 9);
	EXPECT_DOUBLE_EQ(values, 10);

	global::set_generator(old_generator);
}


#endif // DISABLE_TENNCOR_NN_TEST
