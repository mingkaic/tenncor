//
// Created by Mingkai Chen on 2018-01-17.
//

#ifndef TENNCOR_MOCK_ACTOR_H
#define TENNCOR_MOCK_ACTOR_H

#include <algorithm>

#include "include/tensor/tensor_actor.hpp"

using namespace nnet;

template <typename T>
struct mock_actor : public tens_template<T>
{
	mock_actor (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) : tens_template<T>(dest, srcs) {}

	virtual void action (void)
	{
		size_t n_elems = this->dest_.second.n_elems();
		for (size_t i = 0; i < this->srcs_.size(); i++)
		{
			n_elems = std::min(n_elems, this->srcs_[i].second.n_elems());
		}
		for (size_t j = 0; j < n_elems; j++)
		{
			this->dest_.first[j] = 0;
			for (size_t i = 0; i < this->srcs_.size(); i++)
			{
				this->dest_.first[j] += this->srcs_[i].first[j];
			}
		}
	}
};

#endif //TENNCOR_MOCK_ACTOR_H

