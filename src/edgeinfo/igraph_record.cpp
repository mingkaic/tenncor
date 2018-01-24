//
// Created by Ming Kai Chen on 2017-11-08.
//

#include "include/edgeinfo/igraph_record.hpp"

#ifdef IGRAPH_RECORD_HPP

namespace rocnnet_record
{

igraph_record::~igraph_record (void)
{
	record_status::rec_good = false;
}

}

#endif
