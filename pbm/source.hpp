///
/// source.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "llo/node.hpp"

#include "pbm/graph.pb.h"

void save_data (tenncor::Source* out, llo::iSource* in);

llo::DataNode load_source (const tenncor::Source& source);
