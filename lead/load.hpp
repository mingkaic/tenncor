/*!
 *
 *  load.hpp
 *  lead
 *
 *  Purpose:
 *  load graph and data
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "lead/data.pb.h"
#include "lead/graph.pb.h"

#include "kiln/graph.hpp"

#pragma once
#ifndef LEAD_LOAD_HPP
#define LEAD_LOAD_HPP

namespace lead
{

using LeafSetT = std::unordered_set<kiln::Identifier*>;

using RootIds = std::unordered_set<std::string>;

std::unique_ptr<kiln::Graph> load_graph (LeafSetT& leafset, RootIds& rootids,
	const tenncor::GraphPb& ingraph, const tenncor::DataRepoPb& in);

}

#endif /* LEAD_LOAD_HPP */
