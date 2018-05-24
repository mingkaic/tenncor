/*!
 *
 *  identifier.hpp
 *  wire
 *
 *  Purpose:
 *  node proxy to enforce id and labeling functionality,
 *  and safely destroy
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "wire/graph.hpp"

#pragma once
#ifndef WIRE_IDENTIFIER_HPP
#define WIRE_IDENTIFIER_HPP

namespace wire
{

//! wraps iNode and supplies = data
class Identifier
{
public:
	virtual ~Identifier (void);

	Identifier (const Identifier& other);

	Identifier (Identifier&& other);

	Identifier& operator = (const Identifier& other);

	Identifier& operator = (Identifier&& other);

	std::string get_uid (void) const;

	std::string get_label (void) const;

	std::string get_name (void) const;

	bool has_data (void) const
	{
		return arg_->has_data();
	}

	clay::State get_state (void) const
	{
		return arg_->get_state();
	}

	Identifier* derive (Identifier* wrt)
	{
		return new Identifier(graph_, arg_->derive(wrt->arg_.get()),
			this->label_ + "_grad");
	}

protected:
	friend class Functor;

	Identifier (Graph* graph, mold::iNode* arg, std::string label);

	Graph* graph_;

	std::unique_ptr<mold::iNode> arg_;

private:
	friend class Graph;

	std::string label_;

	std::string uid_;
};

}

#endif /* WIRE_IDENTIFIER_HPP */
