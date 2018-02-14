/*!
 *
 *  iconnector.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph connector interface
 *  manages graph information
 *
 *  Created by Mingkai Chen on 2016-12-01.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <queue>

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_ICONNECTOR_HPP
#define TENNCOR_ICONNECTOR_HPP

namespace nnet
{

//! backward transfer function, get gradient nodes; F: Nf -> Nb
using BACKMAP_F = std::function<varptr(std::vector<std::pair<inode*,inode*> >)>;

//! calculate output shape from argument shapes
using SHAPER_F = std::function<tensorshape(std::vector<tensorshape>)>;

class iconnector : public inode, public iobserver
{
public:
	virtual ~iconnector (void);

	//! clone function
	iconnector* clone (void) const;

	//! move function
	iconnector* move (void);

	//! declare copy assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (const iconnector& other);

	//! declare move assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (iconnector&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get unique label with arguments
	virtual std::string get_name (void) const;

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const;

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! get all observerables
	std::vector<inode*> get_arguments (void) const;

protected:
	//! Set dependencies
	iconnector (std::vector<inode*> dependencies, std::string label);

	//! Declare copy constructor to enforce proper g_man_ copy over
	iconnector (const iconnector& other);

	//! Declare move constructor to enforce proper g_man_ copy over
	iconnector (iconnector&& other);



	// >>>>>>>>>>>> KILL CONDITION <<<<<<<<<<<<

	//! suicides when any dependency dies
	virtual void death_on_broken (void)
	{
		delete this;
	}

private:
	void copy_helper (const iconnector& other);

	void move_helper (iconnector&& other);

	//! the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	size_t depth_ = 0;
};

}

#endif /* TENNCOR_ICONNECTOR_HPP */
