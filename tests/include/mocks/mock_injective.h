//
// Created by Mingkai Chen on 2017-03-26.
//

#ifndef TENNCOR_MOCK_IMMUTABLE_H
#define TENNCOR_MOCK_IMMUTABLE_H

#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"

#include "include/graph/connector/immutable/elem_op.hpp"

using namespace nnet;


SHAPER get_testshaper (FUZZ::fuzz_test* fuzzer)
{
	tensorshape shape = random_def_shape(fuzzer);
	return [shape](std::vector<tensorshape>) { return shape; };
}


struct test_actor : public tens_template<double>
{
	test_actor (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs) :
	tens_template(dest, srcs) {}

	virtual void action (void)
	{
		size_t n_elems = this->dest_.second.n_elems();
		std::uniform_real_distribution<double> dist(0, 13);

		auto gen = std::bind(dist, nnutils::get_generator());
		std::generate(this->dest_.first, this->dest_.first + n_elems, gen); // initialize to avoid errors
		if (this->srcs_.size())
		{
			n_elems = std::min(n_elems, this->srcs_[0].second.n_elems());
			std::memcpy(this->dest_.first, this->srcs_[0].first, n_elems * sizeof(double));
		}
	}
};


itens_actor* test_abuilder (out_wrapper<void>& dest,
	std::vector<in_wrapper<void> >& srcs, nnet::TENS_TYPE type)
{
	return new test_actor(dest, srcs);
}


inode* testback (std::vector<std::pair<inode*,inode*> >)
{
	return nullptr;
}


class mock_elem_op : public elem_op
{
public:
	mock_elem_op (std::vector<inode*> args,
		std::string label, SHAPER shapes,
		CONN_ACTOR tfunc = test_abuilder,
		BACK_MAP back = testback) :
	elem_op(args, shapes, new actor_func(tfunc), back, label) {}

	std::function<void(mock_elem_op*)> triggerOnDeath;

	itens_actor*& get_actor (void) { return actor_; }

	virtual void death_on_broken (void)
	{
		if (triggerOnDeath)
		{
			triggerOnDeath(this);
		}
		elem_op::death_on_broken();
	}
};


#endif //TENNCOR_MOCK_IMMUTABLE_H
