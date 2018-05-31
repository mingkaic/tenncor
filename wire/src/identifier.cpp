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
	graph_(graph), arg_(arg), label_(label)
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
	graph_(other.graph_),arg_(other.arg_->clone()),
	label_(other.label_), uid_(graph_->associate(this)) {}

Identifier::Identifier (Identifier&& other) :
	graph_(std::move(other.graph_)), arg_(std::move(other.arg_)),
	label_(std::move(other.label_)), uid_(graph_->associate(this))
{
	graph_->disassociate(other.uid_);
	other.graph_ = nullptr;
}

Identifier& Identifier::operator = (const Identifier& other)
{
	if (this != &other)
	{
		graph_->disassociate(uid_);

		graph_ = other.graph_;
		arg_ = std::unique_ptr<mold::iNode>(other.arg_->clone());
		label_ = other.label_;
		uid_ = graph_->associate(this);
	}
	return *this;
}

Identifier& Identifier::operator = (Identifier&& other)
{
	if (this != &other)
	{
		graph_->disassociate(uid_);

		graph_ = std::move(other.graph_);
		graph_->disassociate(other.uid_);
		other.graph_ = nullptr;
		arg_ = std::move(other.arg_);
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

}

#endif
