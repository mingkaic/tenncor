//
// Created by Mingkai Chen on 2017-04-19.
//

#ifndef TENNCOR_MOCK_LEAF_H
#define TENNCOR_MOCK_LEAF_H

#include "gtest/gtest.h"

#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"

#include "include/graph/leaf/ileaf.hpp"

using namespace nnet;


class mock_leaf : public ileaf
{
public:
	mock_leaf (FUZZ::fuzz_test* fuzzer, std::string name) : ileaf(random_def_shape(fuzzer), name) {}
	mock_leaf (const tensorshape& shape, std::string name) : ileaf(shape, name) {}

	virtual varptr derive (inode*) { return nullptr; }

	void set_good (void) { this->is_init_ = true; }
	void mock_init_data (initializer& initer) { initer(*this->data_); }

protected:
	virtual inode* clone_impl (void) const { return new mock_leaf(*this); }
	virtual inode* move_impl (void) { return new mock_leaf(std::move(*this)); }
	virtual inode* get_gradient (variable*) { return nullptr; }
};


#endif //TENNCOR_MOCK_LEAF_H
