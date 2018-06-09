//
//  identifier.cpp
//  kiln
//

#include <cassert>

#include "ioutil/stream.hpp"

#include "kiln/identifier.hpp"

#include "mold/iobserver.hpp"

#ifdef KILN_IDENTIFIER_HPP

namespace kiln
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
	clear();
}

Identifier::Identifier (const Identifier& other)
{
	copy_helper(other);
}

Identifier::Identifier (Identifier&& other)
{
	move_helper(std::move(other));
}

Identifier& Identifier::operator = (const Identifier& other)
{
	if (this != &other)
	{
		copy_helper(other);
	}
	return *this;
}

Identifier& Identifier::operator = (Identifier&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}

UID Identifier::get_uid (void) const
{
	return uid_;
}

std::string Identifier::get_label (void) const
{
	return label_;
}

std::string Identifier::get_name (void) const
{
	return ioutil::Stream() << "<" << label_
		<< ":" << uid_ << ">";
}

bool Identifier::has_data (void) const
{
	return get()->has_data();
}

clay::State Identifier::get_state (void) const
{
	return get()->get_state();
}

void Identifier::copy_helper (const Identifier& other)
{
	clear();

	graph_ = other.graph_;
	death_sink_ = new mold::OnDeath(
		other.get()->clone(),
		[this]()
		{
			this->death_sink_ = nullptr;
			delete this;
		});
	label_ = other.label_;
	uid_ = graph_->associate(this);
}

void Identifier::move_helper (Identifier&& other)
{
	clear();

	graph_ = std::move(other.graph_);
	graph_->disassociate(other.uid_);
	other.graph_ = nullptr;
	death_sink_ = new mold::OnDeath(
		std::move(*other.death_sink_),
		[this]()
		{
			this->death_sink_ = nullptr;
			delete this;
		});
	other.death_sink_->clear_term();
	delete other.death_sink_;
	other.death_sink_ = nullptr;
	label_ = std::move(other.label_);
	uid_ = graph_->associate(this);
}

void Identifier::clear (void)
{
	if (nullptr != graph_)
	{
		graph_->disassociate(uid_);
	}
	if (nullptr != death_sink_)
	{
		mold::iNode* node = get();
		death_sink_->clear_term();
		delete node;
	}
}

struct Assoc final : public mold::iObserver
{
	Assoc (mold::iNode* source, mold::iNode* kill) :
		mold::iObserver({source}),
		killer_(new mold::OnDeath(kill,
		[this]()
		{
			killer_ = nullptr;
		})) {}

	~Assoc (void)
	{
		if (nullptr != killer_)
		{
			delete killer_->get();
		}
	}

	void initialize (void) override {}

	void update (void) override {}

private:
	mold::OnDeath* killer_;
};

void assoc (Identifier* source, Identifier* kill)
{
	new Assoc(source->get(), kill->get());
}

}

#endif
