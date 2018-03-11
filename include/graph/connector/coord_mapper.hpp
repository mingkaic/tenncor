/*!
 *
 *  coord_mapper.hpp
 *  cnnet
 *
 *  Purpose:
 *  maps source data 1 to 1 with dest data via index map
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/functor.hpp"

#pragma once
#ifndef TENNCOR_COORD_MAPPER_HPP
#define TENNCOR_COORD_MAPPER_HPP

namespace nnet
{

functor* coord_func (inode* arg, SIDX_F smap, USHAPE_F shaper, std::string name, bool same_fb = false);

}

#endif /* TENNCOR_COORD_MAPPER_HPP */
