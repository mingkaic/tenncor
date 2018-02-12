//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/react/iobserver.hpp"

#ifdef TENNCOR_SUBJECT_HPP

#if defined(CSV_RCD) || defined(RPC_RCD)
#include "include/edgeinfo/igraph_record.hpp"
#endif /* CSV_RCD || RPC_RCD */

#ifdef RPC_RCD
#include "include/edgeinfo/rpc_record/grpc_record.hpp"
#endif /* RPC_RCD */

namespace nnet
{

subject::~subject (void)
{
	std::unordered_set<iobserver*> killers = killers_;
	for (iobserver* killer : killers)
	{
		killer->remove_ondeath_dependent(this);
	}

	notify(UNSUBSCRIBE); // unsubscribe all audiences

#if defined(CSV_RCD) || defined(RPC_RCD)

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->node_release(this);
}

#endif /* CSV_RCD || RPC_RCD */
}

subject& subject::operator = (const subject&) { return *this; }

subject& subject::operator = (subject&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}


void subject::notify (notification msg) const
{
#ifdef RPC_RCD
if (rocnnet_record::record_status::rec_good && UPDATE == msg)
{
	if (rocnnet_record::rpc_record* grpc =
		dynamic_cast<rocnnet_record::rpc_record*>(
		rocnnet_record::record_status::rec.get()))
	{
		grpc->data_update(this);
	}
}
#endif /* RPC_RCD */
	for (iobserver* viewer : audience_)
	{
		viewer->update(msg);
	}
}

AUD_SET subject::get_audience (void) const
{
	return audience_;
}


subject::subject (void) {}

void subject::death_on_noparent (void) {}

void subject::attach_killer (iobserver* killer)
{
	if (killer && killers_.end() == killers_.find(killer))
	{
		killers_.insert(killer);
		killer->add_ondeath_dependent(this);
	}
}

void subject::detach_killer (iobserver* killer)
{
	if (killer && killers_.end() != killers_.find(killer))
	{
		killers_.erase(killer);
		killer->remove_ondeath_dependent(this);
	}
}

subject::subject (const subject&) {}

subject::subject (subject&& other)
{
	move_helper(std::move(other));
}

void subject::attach (iobserver* viewer, size_t idx)
{
#if defined(CSV_RCD) || defined(RPC_RCD)

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->edge_capture(viewer, this, idx);
}

#endif /* CSV_RCD || RPC_RCD */

	audience_.emplace(viewer);
}

void subject::detach (iobserver* viewer)
{
#if defined(CSV_RCD) || defined(RPC_RCD)

if (rocnnet_record::record_status::rec_good)
{
	// record subject-object edge
	for (size_t idx : audience_[viewer])
	{
		rocnnet_record::record_status::rec->edge_release(viewer, this, idx);
	}
}

#endif /* CSV_RCD || RPC_RCD */

	audience_.erase(viewer);
	if (audience_.empty())
	{
		death_on_noparent();
	}
}

void subject::detach (iobserver* viewer, size_t idx)
{
#if defined(CSV_RCD) || defined(RPC_RCD)

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->edge_release(viewer, this, idx);
}

#endif /* CSV_RCD || RPC_RCD */

	audience_.erase(viewer);
	if (audience_.empty())
	{
		death_on_noparent();
	}
}

void subject::move_helper (subject&& other)
{
	audience_ = std::move(other.audience_);
	for (iobserver* aud : audience_)
	{
		aud->replace_dependency(this, &other);
	}
}

}

#endif