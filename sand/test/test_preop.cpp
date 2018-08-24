#include "gtest/gtest.h"

#include "sand/preop.hpp"


#ifndef DISABLE_PREOP_TEST


template <typename Iterator>
std::string to_str (Iterator begin, Iterator end)
{
	std::stringstream ss;
	ss << "[";
	if (begin != end)
	{
		ss << (double) *begin;
		for (++begin; begin != end; ++begin)
		{
			ss << "," << (double) *begin;
		}
	}
	ss << "]";
	return ss.str();
}


#define EXPECT_ARREQ(arr, arr2)\
EXPECT_TRUE(std::equal(arr.begin(), arr.end(), arr2.begin())) <<\
"expect list " + to_str(arr.begin(), arr.end()) +\
", got " + to_str(arr2.begin(), arr2.end()) + " instead"


TEST(PREOP, Elem)
{
    Shape shape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;

    ElemPreOperator elem;
    std::string enc = elem.encode();
    EXPECT_THROW(elem({}), std::runtime_error);

    Meta uout = elem({
        Meta{shape, type}
    });

    auto slist = shape.as_list();
    auto gotlist = uout.shape_.as_list();
    EXPECT_ARREQ(slist, gotlist);
    EXPECT_EQ(type, uout.type_);

    Meta bout = elem({
        Meta{shape, type},
        Meta{shape, type},
    });

    auto gotblist = bout.shape_.as_list();
    EXPECT_ARREQ(slist, gotblist);
    EXPECT_EQ(type, bout.type_);

    EXPECT_THROW(elem({
        Meta{shape, type},
        Meta{shape, DTYPE::INT8},
    }), std::runtime_error);

    EXPECT_THROW(elem({
        Meta{shape, DTYPE::INT8},
        Meta{shape, type},
    }), std::runtime_error);

    Shape badshape({5, 6, 3, 4, 7, 8, 9});
    EXPECT_THROW(elem({
        Meta{shape, type},
        Meta{badshape, type},
    }), std::runtime_error);

    EXPECT_THROW(elem({
        Meta{badshape, type},
        Meta{shape, type},
    }), std::runtime_error);

    std::string enc2 = ElemPreOperator().encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());
}


TEST(PREOP, Trans)
{
    Shape shape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;

    EXPECT_THROW(TransPreOperator({0, 1, 1, 9}), std::runtime_error);
    EXPECT_THROW(TransPreOperator({0, 1, 9, 10}), std::runtime_error);
    TransPreOperator trans({0, 1, 1, 2});
    std::vector<DimT> expect = {4, 3, 5, 6, 7, 8, 9};

    std::string enc = trans.encode();

    EXPECT_THROW(trans({}), std::runtime_error);

    EXPECT_THROW(trans({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);

    Meta uout = trans({
        Meta{shape, type}
    });

    auto gotlist = uout.shape_.as_list();
    EXPECT_ARREQ(expect, gotlist);
    EXPECT_EQ(type, uout.type_);

    std::string enc2 = TransPreOperator({0, 1, 1, 2}).encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());

    TransPreOperator trans2({0, 3, 5, 6});
    std::vector<DimT> expect2 = {8, 6, 7, 3, 4, 5, 9};

    Meta uout2 = trans2({
        Meta{shape, type}
    });
    auto gotlist2 = uout2.shape_.as_list();
    EXPECT_ARREQ(expect2, gotlist2);
}


TEST(PREOP, Mat)
{
    Shape ashape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;
    DTYPE badtype = DTYPE::INT32;

    EXPECT_THROW(MatPreOperator({0, 1}, {1, 2}), std::runtime_error);
    MatPreOperator mat({1, 2}, {1, 2});
    Shape bshape({8, 3, 5, 6, 7, 8, 9});
    std::vector<DimT> expect = {8, 4, 5, 6, 7, 8, 9};

    std::string enc = mat.encode();

    EXPECT_THROW(mat({
        Meta{ashape, badtype},
        Meta{bshape, type},
    }), std::runtime_error);
    EXPECT_THROW(mat({
        Meta{ashape, type},
        Meta{bshape, badtype},
    }), std::runtime_error);
    Meta bout = mat({
        Meta{ashape, type},
        Meta{bshape, type},
    });
    auto gotlist = bout.shape_.as_list();
    EXPECT_ARREQ(expect, gotlist);
    EXPECT_EQ(type, bout.type_);

    std::string enc2 = MatPreOperator({1, 2}, {1, 2}).encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());

    MatPreOperator mat2({3, 5}, {3, 6});
    Shape bshape2({12, 11, 10, 3, 4, 5, 8, 9});
    std::vector<DimT> expect2 = {12, 11, 10, 6, 7, 8, 9};

    Meta bout2 = mat2({
        Meta{ashape, type},
        Meta{bshape2, type},
    });
    auto gotlist2 = bout2.shape_.as_list();
    EXPECT_ARREQ(expect2, gotlist2);
    EXPECT_EQ(type, bout2.type_);
}


TEST(PREOP, Typecast)
{
    std::vector<DimT> expect = {3, 4, 5, 6, 7, 8, 9};
    Shape shape(expect);
    DTYPE type = DTYPE::FLOAT;
    DTYPE otype = DTYPE::INT32;

    TypecastPreOperator tcast(otype, type);
    EXPECT_THROW(tcast({}), std::runtime_error);
    EXPECT_THROW(tcast({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);
    EXPECT_THROW(tcast({
        Meta{shape, otype},
    }), std::runtime_error);
    Meta out = tcast({
        Meta{shape, type},
    });

    std::string enc = tcast.encode();

    auto gotlist = out.shape_.as_list();
    EXPECT_ARREQ(expect, gotlist);
    EXPECT_EQ(otype, out.type_);

    std::string enc2 = TypecastPreOperator(otype, type).encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());
}


TEST(PREOP, NElems)
{
    Shape shape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;

    NElemsPreOperator nelems;
    EXPECT_THROW(nelems({}), std::runtime_error);
    EXPECT_THROW(nelems({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);
    Meta out = nelems({
        Meta{shape, type},
    });

    EXPECT_EQ(1, out.shape_.n_elems());
    EXPECT_EQ(1, out.shape_.n_rank());

    EXPECT_EQ(DTYPE::UINT32, out.type_);

    std::string enc = nelems.encode();

    std::string enc2 = NElemsPreOperator().encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());
}


TEST(PREOP, NDims)
{
    Shape shape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;

    NDimsPreOperator all;
    EXPECT_THROW(all({}), std::runtime_error);
    EXPECT_THROW(all({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);
    Meta out = all({
        Meta{shape, type},
    });

    EXPECT_EQ(1, out.shape_.n_elems());
    EXPECT_EQ(1, out.shape_.n_rank());

    EXPECT_EQ(DTYPE::UINT8, out.type_);

    std::string enc = all.encode();

    std::string enc2 = NDimsPreOperator().encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());

    EXPECT_THROW(NDimsPreOperator(8), std::runtime_error);
    NDimsPreOperator dimed(4);
    Meta out2 = dimed({
        Meta{shape, type},
    });

    EXPECT_EQ(7, out2.shape_.n_elems());
    EXPECT_EQ(1, out2.shape_.n_rank());

    EXPECT_EQ(DTYPE::UINT8, out2.type_);

    std::string enc3 = dimed.encode();

    std::string enc4 = NDimsPreOperator(4).encode();
    EXPECT_STREQ(enc3.c_str(), enc4.c_str());
}


TEST(PREOP, Binom)
{
    std::vector<DimT> expect = {3, 4, 5, 6, 7, 8, 9};
    Shape shape(expect);
    Shape badshape({5, 6, 3, 4, 7, 8, 9});
    DTYPE type = DTYPE::INT32;

    BinomPreOperator binom;
    std::string enc = binom.encode();
    EXPECT_THROW(binom({}), std::runtime_error);
    EXPECT_THROW(binom({
        Meta{shape, type},
    }), std::runtime_error);
    EXPECT_THROW(binom({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);
    EXPECT_THROW(binom({
        Meta{shape, DTYPE::DOUBLE},
        Meta{shape, type},
    }), std::runtime_error);
    EXPECT_THROW(binom({
        Meta{badshape, type},
        Meta{shape, DTYPE::DOUBLE}
    }), std::runtime_error);

    Meta out = binom({
        Meta{shape, type},
        Meta{shape, DTYPE::DOUBLE}
    });

    auto gotslist = out.shape_.as_list();
    EXPECT_ARREQ(expect, gotslist);
    EXPECT_EQ(type, out.type_);
}


TEST(PREOP, Reduce)
{
    Shape shape({3, 4, 5, 6, 7, 8, 9});
    DTYPE type = DTYPE::FLOAT;

    ReducePreOperator all;
    EXPECT_THROW(all({}), std::runtime_error);
    EXPECT_THROW(all({
        Meta{shape, type},
        Meta{shape, type},
    }), std::runtime_error);
    Meta out = all({
        Meta{shape, type},
    });

    EXPECT_EQ(1, out.shape_.n_elems());
    EXPECT_EQ(1, out.shape_.n_rank());

    EXPECT_EQ(type, out.type_);

    std::string enc = all.encode();

    std::string enc2 = ReducePreOperator().encode();
    EXPECT_STREQ(enc.c_str(), enc2.c_str());

    EXPECT_THROW(ReducePreOperator(8), std::runtime_error);
    ReducePreOperator dimed(4);
    std::vector<DimT> expect = {3, 4, 5, 6, 8, 9};
    Meta out2 = dimed({
        Meta{shape, type},
    });

    auto gotslist = out2.shape_.as_list();
    EXPECT_ARREQ(expect, gotslist);

    EXPECT_EQ(type, out2.type_);

    std::string enc3 = dimed.encode();

    std::string enc4 = ReducePreOperator(4).encode();
    EXPECT_STREQ(enc3.c_str(), enc4.c_str());
}


#endif /* DISABLE_PREOP_TEST */
