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

struct AssignIO final : public clay::iSource
{
	AssignIO (clay::State state) : state_(state) {}

	bool read_data (clay::State& dest) const override
	{
		bool success = dest.shape_.is_compatible_with(state_.shape_) &&
			dest.dtype_ == state_.dtype_;
		if (success)
		{
			size_t nbytes = state_.shape_.n_elems() * clay::type_size(state_.dtype_);
			std::memcpy(dest.data_.lock().get(), state_.data_.lock().get(), nbytes);
		}
		return success;
	}

private:
	clay::State state_;
};

bool shape_fits (clay::Shape shape, size_t n)
{
	bool compatible = true;
	// perfect fit
	if (shape.is_fully_defined())
	{
		compatible = n == shape.n_elems();
	}
	else
	{
		size_t known = shape.n_known();
		if (0 < known)
		{
			compatible = 0 == n % known;
		}
	}
	return compatible;
}

class Placeholder : public Identifier
{
public:
	Placeholder (std::string label,
		Graph& graph = Graph::get_global()) :
		Identifier(&graph, new mold::Variable(), label,
		[](mold::Variable* var)
		{
			RawBuilder builder;
			var->initialize(builder);
		}) {}

	Placeholder (clay::Shape shape, std::string label,
		Graph& graph = Graph::get_global()) :
		Identifier(&graph, new mold::Variable(), label,
		[shape](mold::Variable* var)
		{
			RawBuilder builder;
			if (shape.is_fully_defined())
			{
				var->initialize(builder, shape);
			}
			else
			{
				var->initialize(builder);
			}
		}) {}

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
		mold::Variable* arg = static_cast<mold::Variable*>(get());
		if (false == arg->has_data())
		{
			graph_->initialize(id->get_uid());
		}
		clay::State state = arg->get_state();
		if (false == shape_fits(state.shape_, n))
		{
			throw std::logic_error(nnutils::formatter() << "data with " 
				<< n << " elements cannot be assigned to allcoated tensor with " 
				<< data_->get_shape().n_elems() << " elements");
		}
		AssignIO assign(state);
		arg->assign(assign);
		
		for (clay::iObserver* aud : audience_)
		{
			aud->update();
		}
		return *this;
	}
};

}

#endif /* WIRE_PLACEHOLDER_HPP */
