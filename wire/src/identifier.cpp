//
//  identifier.cpp
//  wire
//

#include "wire/identifier.hpp"

#ifdef WIRE_IDENTIFIER_HPP

namespace wire
{

mold::TermF bind_id (Identifier* id)
{
	return [id]() {
		id->disassoc(id->uid_);
	};
}

Identifier::Identifier (Graph* graph, mold::iNode* arg, std::string label) :
	OnDeath(arg, bind_id(this)), graph_(graph), label_(label)
{
	uid_ = graph_->associate(this);
}

Identifier::Identifier (Graph* graph, mold::iNode* arg, std::string label,
	InitF init) :
	OnDeath(arg, bind_id(this)), graph_(graph), label_(label)
{
	uid_ = graph_->associate(this);
	graph_->add_uninit(uid_, init);
}

Identifier::~Identifier (void)
{
	// arg must be detached before arg deletion,
	// iObserver has this, but it's executed after this block
	for (mold::iNode*& arg : args_)
	{
		arg->del(this);
	}
	delete get();
	args_.clear();
}

Identifier::Identifier (const Identifier& other) :
	OnDeath(other.args_[0]->clone(), bind_id(this)), graph_(other.graph_),
	label_(other.label_)
{
	for (mold::iNode* arg : args_)
	{
		arg->add(this);
	}
	uid_ = graph_->associate(this);
}

Identifier::Identifier (Identifier&& other) :
	OnDeath(std::move(other), bind_id(this)), graph_(std::move(other.graph_)),
	label_(std::move(other.label_))
{
	graph_->disassociate(other.uid_);
	other.graph_ = nullptr;
	uid_ = graph_->associate(this);
}

Identifier& Identifier::operator = (const Identifier& other)
{
	if (this != &other)
	{
		for (mold::iNode* arg : args_)
		{
			arg->del(this);
		}
		graph_->disassociate(uid_);

		args_[0] = other.args_[0]->clone();
		for (mold::iNode* arg : args_)
		{
			arg->add(this);
		}
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
		mold::OnDeath::operator = (std::move(other));
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

}

#endif
