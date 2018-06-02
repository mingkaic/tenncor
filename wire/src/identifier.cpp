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
	mold::iObserver({arg}), graph_(graph), label_(label)
{
	assert(nullptr != arg && nullptr != graph_);
	uid_ = graph_->associate(this);
}

Identifier::~Identifier (void)
{
	if (graph_)
	{
		graph_->disassociate(uid_);
	}
}

Identifier::Identifier (const Identifier& other) :
	mold::iObserver({other.args_[0]->clone()}), graph_(other.graph_),
	label_(other.label_), uid_(graph_->associate(this)) {}

Identifier::Identifier (Identifier&& other) :
	mold::iObserver(std::move(other)), graph_(std::move(other.graph_)),
	label_(std::move(other.label_)), uid_(graph_->associate(this))
{
	graph_->disassociate(other.uid_);
	other.graph_ = nullptr;
}

Identifier& Identifier::operator = (const Identifier& other)
{
	if (this != &other)
	{
		for (mold::iNode* arg : args_)
		{
			arg->del(this);
		}
		args_ = {other.args_[0]->clone()};
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
		iObserver::operator = (std::move(other));
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
	return args_[0]->has_data();
}

clay::State Identifier::get_state (void) const
{
	return args_[0]->get_state();
}

}

#endif
