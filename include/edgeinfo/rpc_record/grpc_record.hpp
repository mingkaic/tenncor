//
//  grpc_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef RPC_RCD

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include "proto/monitor/grpc_gui.grpc.pb.h"

#include "include/edgeinfo/igraph_record.hpp"
#include "include/edgeinfo/rpc_record/gui_notifier.hpp"
#include "include/thread/stoppable_thread.hpp"
#include "include/graph/connector/iconnector.hpp"

#pragma once
#ifndef grpc_record_HPP
#define grpc_record_HPP

namespace rocnnet_record
{

class rpc_record final : public igraph_record
{
public:
	rpc_record (std::string host, size_t port);

	~rpc_record (void);

	virtual void node_release (const nnet::subject* sub);

	void data_update (const nnet::subject* sub);

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

private:
	nodecache_t node_cache_;

	edgecache_t edge_cache_;

	nnet::stoppable_thread server_;
};

}

#endif /* grpc_record_HPP */

#endif /* RPC_RCD */
