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
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
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

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	ileaf* clone (void) const;

	//! move function
	ileaf* move (void);

	//! declare copy assignment to deep copy over data
	virtual ileaf& operator = (const ileaf& other);

	//! declare move assignment to move over data
	virtual ileaf& operator = (ileaf&& other);

	// >>>> IDENTIFICATION <<<<
	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const;

	// >>>> OBSERVER & OBSERVABLE INFO <<<<
	//! get all observerables: no observables
	virtual std::vector<inode*> get_arguments (void) const;

	//! get the number of observables: 0
	virtual size_t n_arguments (void) const;

	// >>>> FORWARD DATA <<<<
	//! get forward passing value, (pull data if necessary)
	virtual const tensor<double>* eval (void);

	//! utility function: get data shape
	virtual tensorshape get_shape (void) const;

	// >>>> GRAPH STATUS <<<<
	//! merge/update the gradient/leaf info
	virtual std::unordered_set<ileaf*> get_leaves (void) const;

	// >>>> NODE STATUS <<<<
	//! check if data is available
	//! (if the node is initialized)
	virtual bool good_status (void) const;

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto& proto);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! assign initializer
	ileaf (const tensorshape& shape, std::string name);

	//! declare copy constructor to deep copy over data
	ileaf (const ileaf& other);

	//! declare move constructor to move over data
	ileaf (ileaf&& other);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! get forward passing value
	//! return nullptr if leaf is not init
	virtual const tensor<double>* get_eval (void) const;

	//! tensor<double> data
	tensor<double>* data_ = nullptr;

	//! is the tensor<double> initialized?
	//! TRUE = initialized/good,
	//! FALSE = uninitialized/bad
	bool is_init_ = false;

	//! common assignment tensor<double> handler
	assign_func<double> assigner_;

private:
	//! copy helper
	void copy_helper (const ileaf& other);

	//! move helper
	void move_helper (ileaf&& other);
};

}

#endif /* TENNCOR_ILEAF_HPP */
