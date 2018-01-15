//
// Created by Mingkai Chen on 2017-03-26.
//

#ifndef TENNCOR_MOCK_IMMUTABLE_H
#define TENNCOR_MOCK_IMMUTABLE_H

#include "tests/include/util_test.h"
#include "tests/include/fuzz.h"

#include "include/graph/connector/immutable/immutable.hpp"

using namespace nnet;


SHAPER get_testshaper (FUZZ::fuzz_test* fuzzer)
{
	tensorshape shape = random_def_shape(fuzzer);
	return [shape](std::vector<tensorshape>) { return shape; };
}


void testtrans (double* dest, std::vector<const double*> src, nnet::shape_io shape)
{
	size_t n_elems = shape.outs_.n_elems();
	std::uniform_real_distribution<double> dist(0, 13);

	auto gen = std::bind(dist, nnutils::get_generator());
	std::generate(dest, dest + n_elems, gen); // initialize to avoid errors
	if (src.size())
	{
		n_elems = std::min(n_elems, shape.ins_[0].n_elems());
		std::memcpy(dest, src[0], n_elems * sizeof(double));
	}
}


inode* testback (std::vector<std::pair<inode*,inode*>>)
{
	return nullptr;
}


class mock_immutable : public immutable
{
public:
	mock_immutable (std::vector<inode*> args, 
		std::string label, SHAPER shapes,
		TRANSFER_FUNC<double> tfunc = testtrans,
		BACK_MAP back = testback) :
	immutable(args, shapes,
	new transfer_func<double>(tfunc), back, label) {}

	std::function<void(mock_immutable*)> triggerOnDeath;

	virtual void death_on_broken (void)
	{
		if (triggerOnDeath)
		{
			triggerOnDeath(this);
		}
		immutable::death_on_broken();
	}
};


#endif //TENNCOR_MOCK_IMMUTABLE_H
