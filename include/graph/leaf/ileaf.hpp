/*!
 *
 *  ileaf.hpp
 *  cnnet
 *
 *  Purpose:
 *  leaf interface abstractly defines
 *  all pure subject nodes in the graph
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/inode.hpp"

#pragma once
#ifndef TENNCOR_ILEAF_HPP
#define TENNCOR_ILEAF_HPP

namespace nnet
{

class ileaf : public inode
{
public:
	virtual ~ileaf (void);

	//! clone function
	ileaf* clone (void) const;

	//! move function
	ileaf* move (void);

	//! declare copy assignment to deep copy over data
	virtual ileaf& operator = (const ileaf& other);

	//! declare move assignment to move over data
	virtual ileaf& operator = (ileaf&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const;

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<ileaf*> get_leaves (void) const;

protected:
	ileaf (std::string name); // todo: evaluate usefulness

	//! assign initializer
	ileaf (const tensorshape& shape, idata_source* source, std::string name);

	//! declare copy constructor to deep copy over data
	ileaf (const ileaf& other);

	//! declare move constructor to move over data
	ileaf (ileaf&& other);

private:
	//! copy helper
	void copy_helper (const ileaf& other);

	//! move helper
	void move_helper (ileaf&& other);
};

}

#endif /* TENNCOR_ILEAF_HPP */

