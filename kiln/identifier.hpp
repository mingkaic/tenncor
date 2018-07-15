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

struct UIDRange
{
	UID uid_;
	mold::Range range_;
};

bool operator == (const UIDRange& a, const UIDRange& b);

struct UIDRangeHasher
{
	size_t operator () (const UIDRange& id) const
	{
		size_t ssize = sizeof(size_t);
		std::string s(3 * ssize, 0);
		std::memcpy(&s[0], &id.uid_, ssize);
		std::memcpy(&s[0] + ssize,
			&id.range_.lower_, ssize);
		std::memcpy(&s[0] + 2 * ssize,
			&id.range_.upper_, ssize);
		return std::hash<std::string>()(s);
	}
};

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

	virtual std::vector<UIDRange> get_args (void) const;

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
