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

//! observes and takes ownership of arg
class Identifier : private mold::OnDeath
{
public:
	Identifier (Graph* graph, mold::iNode* arg, std::string label);

	Identifier (Graph* graph, mold::iNode* arg, std::string label, InitF init);

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
		return get()->has_data();
	}

	clay::State get_state (void) const
	{
		return get()->get_state();
	}

	Identifier* derive (Identifier* wrt)
	{
		mold::iNode* target = wrt->get();
		return new Identifier(graph_, get()->derive(target), this->label_ + "_grad");
	}

protected:
	Graph* graph_;

	mold::iNode* get_node (void) const
	{
		return get();
	}

private:
	friend mold::TermF bind_id (Identifier* id);

	friend class Graph;

	void disassoc (std::string id)
	{
		if (graph_)
		{
			graph_->disassociate(id);
		}
	}

	std::string label_;

	std::string uid_;
};

}

#endif /* WIRE_IDENTIFIER_HPP */
