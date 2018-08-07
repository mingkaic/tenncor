#include "glass/graph.hpp"
#include "glass/data.pb.h"

#ifndef GLASS_DATA_HPP
#define GLASS_DATA_HPP

void save_data (tenncor::DataRepoPb& out, const Session& in);

void load_data (Session& out, const tenncor::DataRepoPb& data);

#endif /* GLASS_DATA_HPP */
