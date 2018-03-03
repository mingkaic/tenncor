//
// Created by Mingkai Chen on 2017-04-19.
//

#ifndef TENNCOR_MOCK_IVARIABLE_H
#define TENNCOR_MOCK_IVARIABLE_H

#include "tests/unit/include/utils/util_test.hpp"
#include "tests/unit/include/utils/fuzz.h"

#include "include/graph/leaf/ivariable.hpp"

using namespace nnet;


class mock_ivariable : public ivariable
{
public:
	mock_ivariable (const tensorshape& shape,
		initializer* init, std::string name) :
	ivariable(shape, nnet::DOUBLE, init, name) {}
	virtual std::unordered_set<ileaf*> get_leaves (void) const
	{
		return std::unordered_set<ileaf*>{};
	}
	initializer* get_initializer (void) { return static_cast<initializer*>(this->init_); }

protected:
	virtual inode* clone_impl (void) const { return new mock_ivariable(*this); }
	virtual inode* move_impl (void) { return new mock_ivariable(std::move(*this)); }
	virtual inode* get_gradient (variable*) { return nullptr; }
};

#endif //TENNCOR_MOCK_IVARIABLE_H

