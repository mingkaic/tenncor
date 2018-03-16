/*!
 *
 *  variable.hpp
 *  cnnet
 *
 *  Purpose:
 *  define the graph variable implementation
 *
 *  Created by Mingkai Chen on 2017-02-27.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_VARIABLE_HPP
#define TENNCOR_VARIABLE_HPP

namespace nnet
{

class variable final : public inode
{
public:
	//! construct to init zero and one
	variable (const tensorshape& shape,
		std::shared_ptr<data_src> source,
		std::string label);

	//! copy construct to init zero and one
	variable (const variable& other);

	//! move construct to init zero and one
	variable (variable&& other);

	//! clone function
	variable* clone (void) const;

	//! move function
	variable* move (void);

	//! declare copy assignment to copy over initializer
	virtual variable& operator = (const variable& other);

	//! declare move assignment to move over initializer
	virtual variable& operator = (variable&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<const inode*> get_leaves (void) const
	{
		return {this};
	}

	std::shared_ptr<data_src> get_source (void) const
	{
		return src_;
	}



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

	// >>>>>> VARIABLE SPECIAL <<<<<<

	//! initialize data, return true if success
	bool initialize (void);

	//! initialize data using shape,
	//! return true if success
	bool initialize (tensorshape shape);

	//! assign contents of input to this, return true if successful
	bool assign (inode* input, bool notify = true);

protected:
	// >>>>>> SERIALIZATION CONSTRUCTION <<<<<<

	variable (tenncor::variable_proto& proto_src,
		std::string label, std::string uid);

	// >>>>>> SERIALIZATION DATA <<<<<<

	virtual NODE_TYPE node_type (void) const;

	// >>>>>> SERIALIZATION ACTOR <<<<<<

	virtual void serialize_detail (google::protobuf::Any* proto_dest);
	
	friend class graph;

private:
	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	void copy_helper (const variable& other);

	void move_helper (variable&& other);


	std::shared_ptr<data_src> src_; // todo: consider this temporary variable

	//! raw data
	std::unique_ptr<tensor> data_ = nullptr;
};

}

#endif /* TENNCOR_VARIABLE_HPP */
