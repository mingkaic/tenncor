/*!
 *
 *  identifier.hpp
 *  kiln
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

#include "kiln/graph.hpp"

#pragma once
#ifndef KILN_IDENTIFIER_HPP
#define KILN_IDENTIFIER_HPP

namespace kiln
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

	UID get_uid (void) const;

	std::string get_label (void) const;

	std::string get_name (void) const;

	bool has_data (void) const;

	clay::State get_state (void) const;

	virtual std::vector<UID> get_args (void) const;

	mold::iNode* get (void) const;

protected:
	friend class Functor;

	friend void assoc (Identifier* source, Identifier* kill);

	friend Identifier* delta (Identifier* root, Identifier* wrt);

	Identifier (Graph* graph, mold::iNode* arg, std::string label);

	Graph* graph_ = nullptr;

	mold::OnDeath* death_sink_ = nullptr;

private:
	friend class Graph;

	void copy_helper (const Identifier& other);

	void move_helper (Identifier&& other);

	void clear (void);

	std::string label_;

	UID uid_;
};

//! associate two identifiers to delete kill when source dies
void assoc (Identifier* source, Identifier* kill);

}

#endif /* KILN_IDENTIFIER_HPP */
