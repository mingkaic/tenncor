//
//  identifier.cpp
//  wire
//

#include <cassert>

#include "wire/identifier.hpp"

#ifdef WIRE_IDENTIFIER_HPP

namespace wire
{

Identifier::Identifier (Graph* graph, mold::iNode* arg, std::string label) :
	graph_(graph), label_(label)
{
	assert(nullptr != arg && nullptr != graph_);
	death_sink_ = new mold::OnDeath(arg,
	[this]()
	{
		this->death_sink_ = nullptr;
		delete this;
	});
	uid_ = graph_->associate(this);
}

Identifier::~Identifier (void)
{
	if (graph_)
	{
		graph_->disassociate(uid_);
	}
	if (death_sink_ != nullptr)
	{
		mold::iNode* node = get();
		death_sink_->clear_term();
		delete node;
	}
}

Identifier::Identifier (const Identifier& other) :
	graph_(other.graph_), label_(other.label_),
	uid_(graph_->associate(this))
{
	death_sink_ = new mold::OnDeath(
		other.get()->clone(),
		[this]()
		{
			this->death_sink_ = nullptr;
			delete this;
		});
}

Identifier::Identifier (Identifier&& other) :
	graph_(std::move(other.graph_)), label_(std::move(other.label_)),
	uid_(graph_->associate(this))
{
	graph_->disassociate(other.uid_);
	other.graph_ = nullptr;

	death_sink_ = new mold::OnDeath(
		std::move(*other.death_sink_),
		[this]()
		{
			this->death_sink_ = nullptr;
			delete this;
		});
}

Identifier& Identifier::operator = (const Identifier& other)
{
	if (this != &other)
	{
		death_sink_->clear_term();
		delete death_sink_;
		death_sink_ = new mold::OnDeath(
			other.get()->clone(),
			[this]()
			{
				this->death_sink_ = nullptr;
				delete this;
			});
		graph_->disassociate(uid_);

		graph_ = other.graph_;
		label_ = other.label_;
		uid_ = graph_->associate(this);
	}
	return *this;
}

Identifier& Identifier::operator = (Identifier&& other)
{
	if (this != &other)
	{
		death_sink_->clear_term();
		delete death_sink_;
		death_sink_ = new mold::OnDeath(
			std::move(*other.death_sink_),
			[this]()
			{
				this->death_sink_ = nullptr;
				delete this;
			});
		graph_->disassociate(uid_);

		graph_ = std::move(other.graph_);
		graph_->disassociate(other.uid_);
		other.graph_ = nullptr;
		label_ = std::move(other.label_);
		uid_ = graph_->associate(this);
	}
	return *this;
}

std::string Identifier::get_uid (void) const
{
	return uid_;
}

std::string Identifier::get_label (void) const
{
	return label_;
}

std::string Identifier::get_name (void) const
{
	return "<" + label_ + ":" + uid_ + ">";
}

bool Identifier::has_data (void) const
{
	return get()->has_data();
}

clay::State Identifier::get_state (void) const
{
	return get()->get_state();
}

}

#endif
