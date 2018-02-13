/*!
 *
 *  inode.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph variable interface
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/graph.hpp"
#include "include/tensor/tensor.hpp"
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

	//! clone function
	inode* clone (void) const;

	//! move function
	inode* move (void);

	//! declare copy assignment to prevent id_ copy over
	virtual inode& operator = (const inode& other);

	//! declare move assignment to prevent id_ copy over
	virtual inode& operator = (inode&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get the unique hash value
	std::string get_uid (void) const;

	//! get the non-unique label set by user, denoting node purpose
	virtual std::string get_label (void) const;

	//! get beautified summary of name and uid, structure varies for inheritors
	virtual std::string get_name (void) const;

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const = 0;

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<ileaf*> get_leaves (void) const = 0;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	// >>>>>> ID MUTATION <<<<<<

	//! modify label to better describe node purpose
	void set_label (std::string label);

	// >>>>>> FORWARD & BACKWARD <<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void) = 0;

	//! get top-level gradient value, used by root nodes
	virtual varptr derive (inode* wrt) = 0;

protected:
	//! default constructor
	inode (std::string name);

	//! declare copy constructor to prevent id_ copy over
	inode (const inode& other);

	//! declare move constructor to prevent id_ copy over
	inode (inode&& other);



	// >>>>>>>>>>>> POLYMORPHIC CLONERS <<<<<<<<<<<<

	//! clone abstraction function
	virtual inode* clone_impl (void) const = 0;

	//! move abstraction function
	virtual inode* move_impl (void) = 0;

private:
	//! uniquely identifier for this node
	std::string id_ = graph::get().register_node(this);

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

	void clear (void);

// not used
	virtual void update (void); // todo: split iobserver
	
protected:
	// prevent death on broken to avoid double deletion on stack
	virtual void death_on_broken (void) {}
};

//! helper function for exposing node's data
template <typename T>
std::vector<T> expose (inode* var)
{
	std::vector<T> out;
	if (nullptr != var)
	{
		if (tensor* tens = var->get_tensor())
		{
			return expose<T>(tens);
		}
	}
	throw std::exception(); // todo: null var or tensor
}

}

#endif /* TENNCOR_INODE_HPP */
