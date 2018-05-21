/*!
 *
 *  Graph.hpp
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

using InitF = std::function<void(mold::Variable*)>;

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

protected:
	Graph (void) = default;

	friend class Identifier;

	std::string associate (Identifier* id);

	void disassociate (std::string id);

	void add_uninit (std::string uid, InitF init);

private:
	std::string gid_ = puid(this);

	OrderedMap<std::string, Identifier*> adjmap_;

	std::unordered_map<std::string, InitF> uninits_;
};

}

#endif /* WIRE_GRAPH_HPP */
