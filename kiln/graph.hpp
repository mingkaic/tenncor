/*!
 *
 *  graph.hpp
 *  kiln
 *
 *  Purpose:
 *  state of created nodes
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <memory>

#include "mold/variable.hpp"

#include "slip/opcode.hpp"

#include "kiln/omap.hpp"
#include "kiln/rand.hpp"

#pragma once
#ifndef KILN_GRAPH_HPP
#define KILN_GRAPH_HPP

namespace kiln
{

class Identifier;

class Functor;

using UID = size_t;

using FunctorSetT = std::unordered_set<Functor*>;

using OpcodeMapT = std::unordered_map<slip::OPCODE,
	FunctorSetT,slip::EnumHash>;

using BuilderMapT = std::unordered_map<UID,clay::BuildTensorT>;

class Graph
{
public:
	static Graph& get_global (void);

	static std::unique_ptr<Graph> get_temp (void);

	static std::unique_ptr<Graph> get_temp (std::string gid);

	static void replace_global (std::unique_ptr<Graph>&& temp);

	Graph (const Graph&) = delete;

	Graph (Graph&&) = delete;

	Graph& operator = (const Graph&) = delete;

	Graph& operator = (Graph&&) = default;


	std::string get_gid (void) const;

	size_t size (void) const;

	list_it<Identifier*> begin (void);

	list_it<Identifier*> end (void);

	list_const_it<Identifier*> begin (void) const;

	list_const_it<Identifier*> end (void) const;

	bool has_node (UID id) const;

	Identifier* get_node (UID id) const;

	bool replace_id (Identifier* id, UID repl_id);


	FunctorSetT get_func (slip::OPCODE opcode) const;


	void initialize_all (void);

	void initialize (UID id);

	size_t n_uninit (void) const;

protected:
	friend class Identifier;

	friend struct Variable;

	friend class Placeholder;

	friend class Functor;

	Graph (void) : gid_(puid(this)) {}

	Graph (std::string gid) : gid_(gid) {}

	UID associate (Identifier* id);

	void disassociate (UID id);

	void add_func (slip::OPCODE opcode, Functor* func);

	void remove_func (Functor* func);

	BuilderMapT uninits_;

private:
	std::string gid_;

	OrderedMap<UID,Identifier*> adjmap_;

	UID next_uid_ = 0;

	OpcodeMapT funcs_;
};

}

#endif /* KILN_GRAPH_HPP */
