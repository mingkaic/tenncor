/*!
 *
 *  inode.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph variable interface
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor_handler.hpp"
#include "include/tensor/tensor_double.hpp"
#include "include/tensor/tensor_signed.hpp"
#include "include/graph/react/subject.hpp"
#include "include/graph/react/iobserver.hpp"

#pragma once
#ifndef TENNCOR_INODE_HPP
#define TENNCOR_INODE_HPP

namespace nnet
{

class varptr;

class ileaf;

class variable;

class inode : public subject
{
public:
	virtual ~inode (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	inode* clone (void) const;

	//! move function
	inode* move (void);

	//! declare copy assignment to prevent id_ copy over
	virtual inode& operator = (const inode& other);

	//! declare move assignment to prevent id_ copy over
	virtual inode& operator = (inode&& other);

	// >>>> IDENTIFICATION <<<<
	//! get the unique hash value
	std::string get_uid (void) const;

	//! get the non-unique label set by user, denoting node purpose
	virtual std::string get_label (void) const;

	//! get beautified summary of name and uid, structure varies for inheritors
	virtual std::string get_name (void) const;

	//! get name described in summary, defaults to name, may differ for special nodes
	virtual std::string get_summaryid (void) const;

	//! modify label to better describe node purpose
	void set_label (std::string label);

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const = 0;

	// >>>> OBSERVER & OBSERVABLE INFO <<<<
	//! get all observerables
	virtual std::vector<inode*> get_arguments (void) const = 0;

	//! get the number of observables
	virtual size_t n_arguments (void) const = 0;

	//! get all possible observers with specified label
	//! return true if such observer is found
	bool find_audience (std::string label, std::unordered_set<inode*>& audience) const;

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! get forward passing value, (pull data if necessary)
	virtual const itensor* eval (void) = 0;

	//! get top-level gradient value, used by root nodes
	virtual varptr derive (inode* wrt) = 0;

	//! utility function: get forward data shape
	virtual tensorshape get_shape (void) const = 0;

	// >>>> GRAPH STATUS <<<<
	//! merge/update the gradient/leaf info
	virtual std::unordered_set<ileaf*> get_leaves (void) const = 0;

	// >>>> NODE STATUS <<<<
	//! check if data is available
	virtual bool good_status (void) const = 0;

	//! read tensor data from protobuf, may modify current data to provide best information
	//! return true if data is stored successfully
	virtual bool read_proto (const tenncor::tensor_proto& proto) = 0;

	//! store any special-case numerical data using a string key
	void set_metadata (std::string key, size_t value);

	//! propagate special-case data from specified node to this node
	void extract_metadata (inode* n);

	//! check for special-case numerical data
	optional<size_t> get_metadata (std::string key) const;

	tenncor::tensor_proto::tensor_t get_type (void) const
	{
		if (const itensor* result = get_eval())
		{
			return result->get_type();
		}
		return tenncor::tensor_proto::BAD_T;
	}

protected:
	// >>>> CONSTRUCTORS <<<<
	//! default constructor
	inode (std::string name);

	//! declare copy constructor to prevent id_ copy over
	inode (const inode& other);

	//! declare move constructor to prevent id_ copy over
	inode (inode&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone abstraction function
	virtual inode* clone_impl (void) const = 0;

	//! move abstraction function
	virtual inode* move_impl (void) = 0;

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! get forward passing value
	virtual const itensor* get_eval (void) const = 0;

	//! grab operational gradient node, used by other nodes
	//! adds to internal caches if need be
	virtual inode* get_gradient (variable* leaf) = 0;

	//! obtain tensor data from source
	const itensor* take_eval (inode* source) const;

	//! allow inheritants to access source's get_gradient with parameter leaf
	inode* take_gradient (inode* source, variable* leaf) const;

private:
	//! uniquely identifier for this node
	std::string id_ = nnutils::uuid(this);

	//! describes this node's purpose
	std::string label_;

	//! record special-case numerical data
	std::unordered_map<std::string, size_t> metadata_;
};

class varptr : public iobserver
{
public:
	void* operator new (size_t) = delete;

	//! nullptr construction
	varptr (void);

	//! wrap ptr construction
	varptr (inode* ptr);

	virtual ~varptr (void);

	//! assign ptr
	varptr& operator = (inode* other);

	//! implicitly converts to inode*
	operator inode* () const;

	//! dereference overload
	inode& operator * (void) const;

	//! pointer member access overload
	inode* operator -> (void) const;

	//! get inner pointer
	inode* get (void) const;

	virtual void update (std::unordered_set<size_t>);

	void clear (void);

	virtual std::string get_label (void) const;
	
protected:
	virtual void death_on_broken (void);
};

//! helper function for exposing node's data (alternatively: node::get_eval()->expose())
template <typename T>
std::vector<T> expose (inode* var)
{
	if (nullptr == var) return std::vector<T>{};
	const itensor* ten = var->eval();
	std::vector<T> result;
	if (const tensor_double* t = dynamic_cast<const tensor_double*>(ten))
	{
		std::vector<double> res = t->expose();
		result = std::vector<T>(res.begin(), res.end());
	}
	else if (const tensor_signed* t = dynamic_cast<const tensor_signed*>(ten))
	{
		std::vector<signed> res = t->expose();
		result = std::vector<T>(res.begin(), res.end());
	}
	return result;
}

}

#endif /* TENNCOR_INODE_HPP */
