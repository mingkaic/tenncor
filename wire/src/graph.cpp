//
//  graph.cpp
//  wire
//

#include "wire/graph.hpp"
#include "wire/identifier.hpp"

#ifdef WIRE_GRAPH_HPP

namespace wire
{
	
Graph& Graph::get_global (void)
{
	static Graph g;
	return g;
}

std::unique_ptr<Graph> Graph::get_temp (void)
{
	return std::unique_ptr<Graph>(new Graph());
}

void Graph::replace_global (std::unique_ptr<Graph>&& temp)
{
	get_global() = std::move(*temp);
}

std::string Graph::get_gid (void) const
{
	return gid_;
}

bool Graph::has_node (std::string id) const
{
	return adjmap_.has(id);
}

Identifier* Graph::get_node (std::string id) const
{
	Identifier* out = nullptr;
	optional<Identifier*> ider = adjmap_.get(id);
	if ((bool) ider)
	{
		out = *ider;
	}
	return out;
}


void Graph::initialize_all (void)
{
	for (auto upair : uninits_)
	{
		optional<Identifier*> id = adjmap_.get(upair.first);
		if ((bool) id)
		{
			mold::Variable* var = static_cast<mold::Variable*>(
				(*id)->get_node());
			upair.second(var);
		}
		else
		{
			// todo: error handle
		}
	}
	uninits_.clear();
}

void Graph::initialize (std::string id)
{
	optional<Identifier*> ider = adjmap_.get(id);
	auto upair = uninits_.find(id);
	if ((bool) ider && uninits_.end() != upair)
	{
		mold::Variable* var = static_cast<mold::Variable*>(
			(*ider)->get_node());
		upair->second(var);
	}
	else
	{
		// todo: error handle
	}
	uninits_.erase(id);
}

std::string Graph::associate (Identifier* ider)
{
	std::string id = puid(ider);
	if (false == adjmap_.put(id, ider))
	{
		throw std::exception(); // todo: add context
	}
	return id;
}

void Graph::disassociate (std::string id)
{
	if (false == adjmap_.remove(id))
	{
		throw std::exception(); // todo: add context
	}
}

void Graph::add_uninit (std::string uid, InitF init)
{
	uninits_[uid] = init;
}

}

#endif
