//
// Created by Mingkai Chen on 2017-04-19.
//

#ifndef TENNCOR_MOCK_IVARIABLE_H
#define TENNCOR_MOCK_IVARIABLE_H

#include "tests/include/util_test.h"
#include "tests/include/fuzz.h"

#include "include/graph/leaf/ivariable.hpp"

using namespace nnet;


class mock_ivariable : public ivariable
{
public:
	mock_ivariable (const tensorshape& shape,
		initializer<double>* init,
		std::string name) : ivariable(shape, init, name) {}
	virtual std::unordered_set<ileaf*> get_leaves (void) const
	{
		return std::unordered_set<ileaf*>{};
	}
	initializer<double>* get_initializer (void) { return static_cast<initializer<double>*>(this->init_); }

protected:
	virtual inode* clone_impl (void) const { return new mock_ivariable(*this); }
	virtual inode* move_impl (void) { return new mock_ivariable(std::move(*this)); }
	virtual inode* get_gradient (variable*) { return nullptr; }
};

#endif //TENNCOR_MOCK_IVARIABLE_H
