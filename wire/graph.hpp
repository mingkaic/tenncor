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

#include "wire/omap.hpp"
#include "wire/rand.hpp"

#pragma once
#ifndef WIRE_GRAPH_HPP
#define WIRE_GRAPH_HPP

namespace wire
{

class Identifier;

class graph
{
public:
	static graph& get_global (void)
	{
		static graph g;
	}

	static std::unique_ptr<graph> get_temp (void)
	{
		return std::unique_ptr<graph>(new graph());
	}

	static void replace_global (std::unique_ptr<graph>&& temp)
	{
		get_global() = std::move(*temp);
	}

	graph (const graph&) = delete;
	graph (graph&&) = delete;
	graph& operator = (const graph&) = delete;


	std::string get_gid (void) const
	{
		return gid_;
	}

	bool has_node (std::string id) const
	{
		return adjmap_.has(id);
	}

	Identifier* get_identifier (std::string id) const
	{
		Identifier* out = nullptr;
		auto it = idmap_.find(id);
		if (idmap_.end() != it)
		{
			optional<Identifier*> id = adjmap_.get(it->second);
			if ((bool) id)
			{
				out = *id;
			}
		}
		return out;
	}

	Identifier* get_identifier (mold::iNode* node) const
	{
		Identifier* out = nullptr;
		optional<Identifier*> id = adjmap_.get(node);
		if ((bool) id)
		{
			out = *id;
		}
		return out;
	}


	void initialize_all (void)
	{
		for (auto upair : uninits_)
		{
			clay::iBuilder* builder = upair->second;
			auto it = idmap_.find(upair->first);
			if (idmap_.end() != it &&
				(mold::Variable* var = dynamic_cast<mold::Variable*>(it->second)))
			{
				var->initialize(*builder);
			}
			else
			{
				// todo: error handle
			}
		}
		uninits_.clear();
	}

	void initialize (std::string id)
	{
		auto it = idmap_.find(id);
		auto ut = uninits_.find(id);
		if (idmap_.end() != it && uninits_.end() != ut)
		{
			it->second->initialize(*ut->second);
		}
		else
		{
			// todo: error handle
		}
		uninits_.erase(id);
	}

protected:
	graph (void) = default;

	friend class Identifier;

	std::string associate (mold::iNode* arg, Identifier* ider)
	{
		if (false == adjmap_.put(arg, ider))
		{
			throw std::exception(); // todo: add context
		}
		std::string id = puid(arg);
		idmap_[id] = arg;
		return id;
	}

	void disassociate (std::string id)
	{
		auto it = idmap_.find(id);
		if (idmap_.end() == it)
		{
			throw std::exception(); // todo: add context
		}
		iNode* del = *it;
		if (false == adjmap_.remove(del))
		{
			throw std::exception();
		}
		idmap_.erase(it);
	}

private:
	std::string gid_ = puid(this);

	OrderedMap<mold::iNode*, Identifier*> adjmap_;

	std::unordered_map<std::string, mold::iNode*> idmap_;

	std::unordered_map<std::string, clay::iBuilder*> uninits_;
};

}

#endif /* WIRE_GRAPH_HPP */
