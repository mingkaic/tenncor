/*!
 *
 *  packer.hpp
 *  lead
 *
 *  Purpose:
 *  abstract builder that validates type and shape
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "lead/data.pb.h"

#pragma once
#ifndef LEAD_PACKER_HPP
#define LEAD_PACKER_HPP

namespace lead
{

std::shared_ptr<char> unpack_data (const google::protobuf::Any& src,
	tenncor::TensorT dtype);

void pack_data (google::protobuf::Any* dest, std::shared_ptr<char> src, size_t n, tenncor::TensorT dtype);

}

#endif /* LEAD_PACKER_HPP */
