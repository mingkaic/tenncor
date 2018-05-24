/*!
 *
 *  graph.hpp
 *  wire
 *
 *  Purpose:
 *  state of created nodes
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <memory>

#include "clay/ibuilder.hpp"
#include "mold/variable.hpp"

#include "wire/omap.hpp"
#include "wire/rand.hpp"

#pragma once
#ifndef WIRE_GRAPH_HPP
#define WIRE_GRAPH_HPP

namespace wire
{

class Identifier;

class Graph
{
public:
	static Graph& get_global (void);

	static std::unique_ptr<Graph> get_temp (void);

	static void replace_global (std::unique_ptr<Graph>&& temp);

	Graph (const Graph&) = delete;
	Graph (Graph&&) = delete;
	Graph& operator = (const Graph&) = delete;

	Graph& operator = (Graph&&) = default;


	std::string get_gid (void) const;

	bool has_node (std::string id) const;

	Identifier* get_node (std::string id) const;


	void initialize_all (void);

	void initialize (std::string id);

	size_t n_uninit (void) const
	{
		return uninits_.size();
	}

protected:
	friend class Identifier;

	friend struct Variable;

	friend class Placeholder;

	Graph (void) = default;

	std::string associate (Identifier* id);

	void disassociate (std::string id);

	std::unordered_map<std::string, std::unique_ptr<clay::iBuilder>> uninits_;

	std::unordered_map<std::string, clay::Shape> alloweds_;

private:
	void unsafe_init (Identifier* id, clay::iBuilder& builder);

	std::string gid_ = puid(this);

	OrderedMap<std::string, Identifier*> adjmap_;
};

}

#endif /* WIRE_GRAPH_HPP */
