/*!
 *
 *  placeholder.hpp
 *  wire
 *
 *  Purpose:
 *  extend variable with vector assignment
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "ioutil/stream.hpp"
#include "mold/variable.hpp"

#include "wire/graph.hpp"
#include "wire/identifier.hpp"

#pragma once
#ifndef WIRE_PLACEHOLDER_HPP
#define WIRE_PLACEHOLDER_HPP

namespace wire
{

bool shape_fits (clay::Shape shape, size_t n);

class Placeholder : public Identifier
{
public:
	Placeholder (std::string label,
		Graph& graph = Graph::get_global());

	Placeholder (clay::Shape shape, std::string label,
		Graph& graph = Graph::get_global());

	Placeholder& operator = (const Placeholder&) = default;

	Placeholder& operator = (Placeholder&& other) = default;

	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	template <typename T>
	Placeholder& operator = (std::vector<T> data)
	{
		size_t n = data.size();
		mold::Variable* arg = static_cast<mold::Variable*>(arg_.get());
		if (false == arg->has_data())
		{
			clay::DTYPE dtype = clay::get_type<T>();
			graph_->initialize(get_uid(), [n, dtype](clay::iBuilder& b)
			{
				RawBuilder* rb = static_cast<RawBuilder*>(&b);

				rb->limit_ = n;
				rb->dtype_ = dtype;
			});
		}
		clay::State state = arg->get_state();
		if (false == shape_fits(state.shape_, n))
		{
			throw std::logic_error(ioutil::Stream() << "data with " 
				<< n << " elements cannot be assigned to allcoated tensor with " 
				<< state.shape_.n_elems() << " elements");
		}
		AssignIO assign(state);
		arg->assign(assign);
		return *this;
	}

private:
	struct AssignIO final : public clay::iSource
	{
		AssignIO (clay::State state);

		bool read_data (clay::State& dest) const override;

	private:
		clay::State state_;
	};

	struct RawBuilder : public clay::iBuilder
	{
		clay::TensorPtrT get (void) const override;

		clay::TensorPtrT get (clay::Shape shape) const override;

		size_t limit_;
		clay::DTYPE dtype_;

	protected:
		clay::iBuilder* clone_impl (void) const override
		{
			return new RawBuilder(*this);
		}
	};
};

}

#endif /* WIRE_PLACEHOLDER_HPP */
