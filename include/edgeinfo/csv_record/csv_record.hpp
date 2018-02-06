//
//  csv_record.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-01-30.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef CSV_RCD

#include <unordered_set>
#include <fstream>
#include <memory>

#include "include/edgeinfo/adjlist_record.hpp"
#include "include/graph/connector/iconnector.hpp"

#pragma once
#ifndef CSV_RECORD_HPP
#define CSV_RECORD_HPP

namespace rocnnet_record
{

class csv_record final : public adjlist_record
{
public:
	csv_record (std::string fname);

	void to_csv (const nnet::iconnector* consider_graph = nullptr) const;

	void setVerbose (bool verbosity);

	void setDisplayShape (bool display);

private:
	bool verbose_ = false;

	bool display_shape_ = true;

	std::string outname_;
};

}

#endif /* CSV_RECORD_HPP */

#endif /* CSV_RCD */
