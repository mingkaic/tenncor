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

#include "mold/ondeath.hpp"

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

	bool has_data (void) const;

	clay::State get_state (void) const;

	virtual Identifier* derive (Identifier* wrt) = 0;

protected:
	friend class Functor;

	Identifier (Graph* graph, mold::iNode* arg, std::string label);

	mold::iNode* get (void) const
	{
		return death_sink_->get();
	}

	Graph* graph_ = nullptr;

	mold::OnDeath* death_sink_ = nullptr;

private:
	friend class Graph;

	void copy_helper (const Identifier& other);

	void move_helper (Identifier&& other);

	void clear (void);

	std::string label_;

	std::string uid_;
};

}

#endif /* WIRE_IDENTIFIER_HPP */
