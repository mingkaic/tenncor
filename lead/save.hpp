/*!
 *
 *  save.hpp
 *  lead
 *
 *  Purpose:
 *  save graph and data
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "lead/data.pb.h"
#include "lead/graph.pb.h"

#include "kiln/graph.hpp"

#pragma once
#ifndef LEAD_SAVE_HPP
#define LEAD_SAVE_HPP

namespace lead
{

void save_tensor (tenncor::TensorPb& out, clay::State state);

void save_data (tenncor::DataRepoPb& out, const kiln::Graph& graph);

void save_graph (tenncor::GraphPb& out, const kiln::Graph& graph);

}

#endif /* LEAD_SAVE_HPP */
