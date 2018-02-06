//
//  igraph_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#if defined(CSV_RCD) || defined(RPC_RCD)

#include "include/graph/inode.hpp"
#include "include/utils/utils.hpp"

#pragma once
#ifndef IGRAPH_RECORD_HPP
#define IGRAPH_RECORD_HPP

namespace rocnnet_record
{

class igraph_record
{
public:
	virtual ~igraph_record (void);

	// all nodes are subjects
	virtual void node_release (const nnet::subject* sub) = 0;

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx) = 0;

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx) = 0;
};

struct record_status
{
	static std::unique_ptr<igraph_record> rec;
	static bool rec_good;
};

}

#endif /* IGRAPH_RECORD_HPP */

#endif /* CSV_RCD || RPC_RCD */
