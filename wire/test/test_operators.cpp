#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "kiln/unif_init.hpp"

#include "wire/operators.hpp"
#include "wire/variable.hpp"
#include "wire/delta.hpp"


#ifndef DISABLE_OPERATORS_TEST


static const double ERR_THRESH = 0.001; // 0.1% error


using namespace testutil;


class OPERATORS : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutil::fuzz_test::TearDown();
		wire::Graph& g = wire::Graph::get_global();
		assert(0 == g.size());
	}
};


using VARFUNC = std::function<wire::Identifier*(std::vector<wire::Identifier*>)>;


using SCALAR = std::function<double(double)>;


template <typename T=double>
using BINAR = std::function<T(T,T)>;


static void selfGrad (wire::Identifier* f, clay::State state)
{
	clay::Shape shape = f->get_state().shape_;
	wire::Identifier* wun = wire::delta(f, f);
	clay::State back = wun->get_state();
	EXPECT_SHAPEQ(shape, back.shape_);
	EXPECT_EQ(state.dtype_, back.dtype_);
	size_t n = shape.n_elems();
	switch (state.dtype_)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* oneptr = (double*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* oneptr = (float*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* oneptr = (int8_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* oneptr = (uint8_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* oneptr = (int16_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* oneptr = (uint16_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* oneptr = (int32_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* oneptr = (uint32_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* oneptr = (int64_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* oneptr = (uint64_t*) back.data_.lock().get();
			for (size_t i = 0; i < n; ++i)
			{
				EXPECT_EQ(1, oneptr[i]) << i;
			}
		}
		default:
		EXPECT_TRUE(false) << "bad grad type";
		break;
	};
}


static void unarElemTest (fuzz_test* fuzzer, VARFUNC op,
	SCALAR expect, SCALAR grad, std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer, {2, 6});
	clay::BuildTensorT builder = kiln::unif_init(
		limits.first, limits.second, shape);
	wire::Variable leaf(builder, "leaf");

	// test behavior G000
	wire::Identifier* f = op({&leaf});
	wire::Identifier* f2 = op({&leaf});
	EXPECT_EQ(f, f2);

	// test behavior G001
	EXPECT_EQ(nullptr, op({nullptr}));

	// test behavior G002
	EXPECT_THROW(wire::delta(f, f), std::exception);
	EXPECT_THROW(wire::delta(f, &leaf), std::exception);

	wire::Graph::get_global().initialize_all();
	ASSERT_TRUE(f->has_data());
	clay::State state = f->get_state();

	// test behavior G003
	selfGrad(f, state);

	// test behavior B1xx
	wire::Identifier* der = wire::delta(f, &leaf);
	clay::State back = der->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_SHAPEQ(shape, back.shape_);
	double* inptr = (double*) leaf.get_state().data_.lock().get();
	ASSERT_FALSE(state.data_.expired());
	ASSERT_FALSE(back.data_.expired());
	double* outptr = (double*) state.data_.lock().get();
	double* backptr = (double*) back.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		double expectfwd = expect(inptr[i]);
		double expectbwd = grad(inptr[i]);
		double err = expectfwd - outptr[i];
		double err2 = expectbwd - backptr[i];
		if (2 < expectfwd)
		{
			err /= expectfwd;
		}
		if (2 < expectbwd)
		{
			err2 /= expectbwd;
		}
		EXPECT_GT(ERR_THRESH, err) << expectfwd << " " << outptr[i];
		EXPECT_GT(ERR_THRESH, err2) << expectbwd << " " << backptr[i];
	}
}


static void binaryElemTest (fuzz_test* fuzzer, VARFUNC op,
	BINAR<double> expect, BINAR<double> gradA, BINAR<double> gradB,
	std::pair<double,double> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer, {2, 6});
	clay::BuildTensorT builder = kiln::unif_init(
		limits.first, limits.second, shape);
	wire::Variable leaf(builder, "leaf");
	wire::Variable leaf2(builder, "leaf2");

	// test behavior G000
	wire::Identifier* f = op({&leaf, &leaf2});
	wire::Identifier* f2 = op({&leaf, &leaf2});
	EXPECT_EQ(f, f2);

	// test behavior G001
	EXPECT_EQ(nullptr, op({&leaf, nullptr}));
	EXPECT_EQ(nullptr, op({nullptr, &leaf2}));
	EXPECT_EQ(nullptr, op({nullptr, nullptr}));

	// test behavior G002
	EXPECT_THROW(wire::delta(f, f), std::exception);
	EXPECT_THROW(wire::delta(f, &leaf), std::exception);
	EXPECT_THROW(wire::delta(f, &leaf2), std::exception);

	wire::Graph::get_global().initialize_all();
	ASSERT_TRUE(f->has_data());
	clay::State state = f->get_state();

	// test behavior G003
	selfGrad(f, state);

	// test behavior B1xx
	wire::Identifier* der = wire::delta(f, &leaf);
	wire::Identifier* der2 = wire::delta(f, &leaf2);
	clay::State back = der->get_state();
	clay::State back2 = der2->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_SHAPEQ(shape, back.shape_);
	EXPECT_SHAPEQ(shape, back2.shape_);
	double* inptr = (double*) leaf.get_state().data_.lock().get();
	double* inptr2 = (double*) leaf2.get_state().data_.lock().get();
	ASSERT_FALSE(state.data_.expired());
	ASSERT_FALSE(back.data_.expired());
	ASSERT_FALSE(back2.data_.expired());
	double* outptr = (double*) state.data_.lock().get();
	double* backptr = (double*) back.data_.lock().get();
	double* backptr2 = (double*) back2.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		double expectfwd = expect(inptr[i], inptr2[i]);
		double expectbwdA = gradA(inptr[i], inptr2[i]);
		double expectbwdB = gradB(inptr[i], inptr2[i]);
		double err = expectfwd - outptr[i];
		double err2 = expectbwdA - backptr[i];
		double err3 = expectbwdB - backptr2[i];
		if (2 < expectfwd)
		{
			err /= expectfwd;
		}
		if (2 < expectbwdA)
		{
			err2 /= expectbwdA;
		}
		if (2 < expectbwdB)
		{
			err3 /= expectbwdB;
		}
		EXPECT_GT(ERR_THRESH, err) << expectfwd << " " << outptr[i];
		EXPECT_GT(ERR_THRESH, err2) << expectbwdA << " " << backptr[i];
		EXPECT_GT(ERR_THRESH, err3) << expectbwdB << " " << backptr2[i];
	}
}


static void binaryIntElemTest (fuzz_test* fuzzer, VARFUNC op,
	BINAR<int16_t> expect, BINAR<int16_t> gradA, BINAR<int16_t> gradB,
	std::pair<int16_t,int16_t> limits = {-1, 1})
{
	clay::Shape shape = random_def_shape(fuzzer, {2, 6});
	clay::BuildTensorT builder = kiln::unif_init(
		limits.first, limits.second, shape);
	wire::Variable leaf(builder, "leaf");
	wire::Variable leaf2(builder, "leaf2");

	// test behavior G000
	wire::Identifier* f = op({&leaf, &leaf2});
	wire::Identifier* f2 = op({&leaf, &leaf2});
	EXPECT_EQ(f, f2);

	// test behavior G001
	EXPECT_EQ(nullptr, op({&leaf, nullptr}));
	EXPECT_EQ(nullptr, op({nullptr, &leaf2}));
	EXPECT_EQ(nullptr, op({nullptr, nullptr}));

	// test behavior G002
	EXPECT_THROW(wire::delta(f, f), std::exception);
	EXPECT_THROW(wire::delta(f, &leaf), std::exception);
	EXPECT_THROW(wire::delta(f, &leaf2), std::exception);

	wire::Graph::get_global().initialize_all();
	ASSERT_TRUE(f->has_data());
	clay::State state = f->get_state();

	// test behavior G003
	selfGrad(f, state);

	// test behavior B1xx
	wire::Identifier* der = wire::delta(f, &leaf);
	wire::Identifier* der2 = wire::delta(f, &leaf2);
	clay::State back = der->get_state();
	clay::State back2 = der2->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_SHAPEQ(shape, back.shape_);
	EXPECT_SHAPEQ(shape, back2.shape_);
	int16_t* inptr = (int16_t*) leaf.get_state().data_.lock().get();
	int16_t* inptr2 = (int16_t*) leaf2.get_state().data_.lock().get();
	ASSERT_FALSE(state.data_.expired());
	ASSERT_FALSE(back.data_.expired());
	ASSERT_FALSE(back2.data_.expired());
	int16_t* outptr = (int16_t*) state.data_.lock().get();
	int16_t* backptr = (int16_t*) back.data_.lock().get();
	int16_t* backptr2 = (int16_t*) back2.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		EXPECT_EQ(expect(inptr[i], inptr2[i]), outptr[i]);
		EXPECT_EQ(gradA(inptr[i], inptr2[i]), backptr[i]);
		EXPECT_EQ(gradB(inptr[i], inptr2[i]), backptr2[i]);
	}
}


TEST_F(OPERATORS, Abs_G0xxAndG100)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::abs(args[0]);
	},
	[](double arg)
	{
		return std::abs(arg);
	},
	[](double)
	{
		return 1;
	});
}


TEST_F(OPERATORS, Neg_G0xxAndG101)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::neg(args[0]);
	},
	[](double arg)
	{
		return -arg;
	},
	[](double)
	{
		return -1;
	});
}


TEST_F(OPERATORS, Not_G0xxAndG102)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::logical_not(args[0]);
	},
	[](double arg)
	{
		return !arg;
	},
	[](double)
	{
		return 0;
	});
}


TEST_F(OPERATORS, Sin_G0xxAndG103)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::sin(args[0]);
	},
	[](double arg)
	{
		return std::sin(arg);
	},
	[](double arg)
	{
		return std::cos(arg);
	});
}


TEST_F(OPERATORS, Cos_G0xxAndG104)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::cos(args[0]);
	},
	[](double arg)
	{
		return std::cos(arg);
	},
	[](double arg)
	{
		return -std::sin(arg);
	});
}


TEST_F(OPERATORS, Tan_G0xxAndG105)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::tan(args[0]);
	},
	[](double arg)
	{
		return std::tan(arg);
	},
	[](double arg)
	{
		double denom = std::cos(arg);
		return 1 / (denom * denom);
	});
}


TEST_F(OPERATORS, Exp_G0xxAndG106)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::exp(args[0]);
	},
	[](double arg)
	{
		return std::exp(arg);
	},
	[](double arg)
	{
		return std::exp(arg);
	});
}


TEST_F(OPERATORS, Log_G0xxAndG107)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::log(args[0]);
	},
	[](double arg)
	{
		return std::log(arg);
	},
	[](double arg)
	{
		return 1 / arg;
	}, {0.5, 7});
}


TEST_F(OPERATORS, Sqrt_G0xxAndG108)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::sqrt(args[0]);
	},
	[](double arg)
	{
		return std::sqrt(arg);
	},
	[](double arg)
	{
		double denom = std::sqrt(arg);
		return 1 / (2 * denom);
	}, {0, 7});
}


TEST_F(OPERATORS, Round_G0xxAndG109)
{
	unarElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::round(args[0]);
	},
	[](double arg)
	{
		return std::round(arg);
	},
	[](double)
	{
		return 1;
	});
}


TEST_F(OPERATORS, Pow_G0xxAndG130)
{
	binaryElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::pow(args[0], args[1]);
	},
	[](double a, double b)
	{
		return std::pow(a, b);
	},
	[](double a, double b)
	{
		return b * std::pow(a, b - 1);
	},
	[](double a, double b)
	{
		return std::pow(a, b) * std::log(a);
	}, {0, 7});
}


TEST_F(OPERATORS, Add_G0xxAndG131)
{
	binaryElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::add(args[0], args[1]);
	},
	[](double a, double b)
	{
		return a + b;
	},
	[](double a, double b)
	{
		return 1;
	},
	[](double a, double b)
	{
		return 1;
	});
}


TEST_F(OPERATORS, Sub_G0xxAndG132)
{
	binaryElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::sub(args[0], args[1]);
	},
	[](double a, double b)
	{
		return a - b;
	},
	[](double a, double b)
	{
		return 1;
	},
	[](double a, double b)
	{
		return -1;
	});
}


TEST_F(OPERATORS, Mul_G0xxAndG133)
{
	binaryElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::mul(args[0], args[1]);
	},
	[](double a, double b)
	{
		return a * b;
	},
	[](double a, double b)
	{
		return b;
	},
	[](double a, double b)
	{
		return a;
	});
}


TEST_F(OPERATORS, Div_G0xxAndG134)
{
	binaryElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::div(args[0], args[1]);
	},
	[](double a, double b)
	{
		return a / b;
	},
	[](double a, double b)
	{
		return 1/b;
	},
	[](double a, double b)
	{
		return -a/std::pow(b, 2);
	}, {0.1, 7});
}


TEST_F(OPERATORS, Eq_G0xxAndG135)
{
	BINAR<int16_t> bin = [](int16_t a, int16_t b)
	{
		return a == b;
	};
	binaryIntElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::eq(args[0], args[1]);
	}, bin, bin, bin);
}


TEST_F(OPERATORS, Neq_G0xxAndG136)
{
	BINAR<int16_t> bin = [](int16_t a, int16_t b)
	{
		return a != b;
	};
	binaryIntElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::neq(args[0], args[1]);
	}, bin, bin, bin);
}


TEST_F(OPERATORS, Lt_G0xxAndG137)
{
	BINAR<int16_t> bin = [](int16_t a, int16_t b)
	{
		return a < b;
	};
	binaryIntElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::lt(args[0], args[1]);
	}, bin, bin, bin);
}


TEST_F(OPERATORS, Gt_G0xxAndG138)
{
	BINAR<int16_t> bin = [](int16_t a, int16_t b)
	{
		return a > b;
	};
	binaryIntElemTest(this,
	[](std::vector<wire::Identifier*> args)
	{
		return wire::gt(args[0], args[1]);
	}, bin, bin, bin);
}


TEST_F(OPERATORS, Matmul_G0xxAndG139)
{
}


#endif /* DISABLE_OPERATORS_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */