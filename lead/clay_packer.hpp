/*!
 *
 *  builder.hpp
 *  kiln
 *
 *  Purpose:
 *  abstract builder that validates type and shape
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "lead/clay.pb.h"

#pragma once
#ifndef LEAD_CLAY_PACKER_HPP
#define LEAD_CLAY_PACKER_HPP

namespace lead
{

std::shared_ptr<char> unpack_data (const google::protobuf::Any& data, 
	TensorT dtype);

void pack_data (std::shared_ptr<char> src, size_t n, TensorT dtype, 
	google::protobuf::Any& data);

}

#endif /* LEAD_CLAY_PACKER_HPP */
