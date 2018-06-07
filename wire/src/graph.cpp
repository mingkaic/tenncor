//
//  graph.cpp
//  wire
//

#include "wire/graph.hpp"
#include "wire/identifier.hpp"
#include "wire/error.hpp"

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

std::unique_ptr<Graph> Graph::get_temp (std::string gid)
{
	return std::unique_ptr<Graph>(new Graph(gid));
}

void Graph::replace_global (std::unique_ptr<Graph>&& temp)
{
	get_global() = std::move(*temp);
}

std::string Graph::get_gid (void) const
{
	return gid_;
}

size_t Graph::size (void) const
{
	return adjmap_.size();
}

list_it<Identifier*> Graph::begin (void)
{
	return adjmap_.begin();
}

list_it<Identifier*> Graph::end (void)
{
	return adjmap_.end();
}

list_const_it<Identifier*> Graph::begin (void) const
{
	return adjmap_.begin();
}

list_const_it<Identifier*> Graph::end (void) const
{
	return adjmap_.end();
}

bool Graph::has_node (UID id) const
{
	return adjmap_.has(id);
}

Identifier* Graph::get_node (UID id) const
{
	Identifier* out = nullptr;
	optional<Identifier*> ider = adjmap_.get(id);
	if ((bool) ider)
	{
		out = *ider;
	}
	return out;
}

bool Graph::replace_id (wire::Identifier* id, UID repl_id)
{
	UID src_id = id->get_uid();
	bool success = adjmap_.remove(src_id) &&
		false == adjmap_.has(repl_id);
	if (success)
	{
		id->uid_ = repl_id;
		adjmap_.put(repl_id, id);

		auto uit = uninits_.find(src_id);
		if (uninits_.end() != uit)
		{
			uninits_[repl_id] = std::move(uit->second);
			uninits_.erase(uit);
		}
		auto sit = alloweds_.find(src_id);
		if (alloweds_.end() != sit)
		{
			alloweds_[repl_id] = sit->second;
			alloweds_.erase(sit);
		}
	}
	return success;
}

FunctorSetT Graph::get_func (slip::OPCODE opcode) const
{
	auto it = funcs_.find(opcode);
	if (funcs_.end() != it)
	{
		return it->second;
	}
	return FunctorSetT();
}

void Graph::initialize_all (void)
{
	for (auto& upair : uninits_)
	{
		optional<Identifier*> id = adjmap_.get(upair.first);
		if ((bool) id)
		{
			unsafe_init(*id, *upair.second.get());
		}
		else
		{
			// todo: error handle
		}
	}
	uninits_.clear();
}

void Graph::initialize (UID id)
{
	optional<Identifier*> ider = adjmap_.get(id);
	auto upair = uninits_.find(id);
	if ((bool) ider && uninits_.end() != upair)
	{
		unsafe_init(*ider, *(upair->second.get()));
	}
	else
	{
		// todo: error handle
	}
	uninits_.erase(id);
}

size_t Graph::n_uninit (void) const
{
	return uninits_.size();
}

UID Graph::associate (Identifier* ider)
{
	UID id = next_uid_++;
	if (false == adjmap_.put(id, ider))
	{
		throw DuplicateNodeIDError(gid_, id);
	}
	return id;
}

void Graph::disassociate (UID id)
{
	if (false == adjmap_.remove(id))
	{
		throw MissingNodeError(gid_, id);
	}
}

void Graph::add_func (slip::OPCODE opcode, Functor* func)
{
	funcs_[opcode].emplace(func);
}

void Graph::remove_func (Functor* func)
{
	for (auto& fpair : funcs_)
	{
		auto it = fpair.second.find(func);
		if (fpair.second.end() != it)
		{
			fpair.second.erase(it);
			return;
		}
	}
}

void Graph::unsafe_init (Identifier* id, clay::iBuilder& builder)
{
	mold::Variable* var = static_cast<mold::Variable*>(id->get());
	clay::Shape allowed;
	auto it = alloweds_.find(id->get_uid());
	if (alloweds_.end() != it)
	{
		allowed = it->second;
		alloweds_.erase(it);
	}
	if (allowed.is_fully_defined())
	{
		var->initialize(builder, allowed);
	}
	else
	{
		var->initialize(builder);
	}
}

}

#endif
