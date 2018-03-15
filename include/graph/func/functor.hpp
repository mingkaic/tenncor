/*!
 *
 *  functor.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph functor operates on nodes
 *
 *  Created by Mingkai Chen on 2016-12-01.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/react/iobserver.hpp"
#include "include/graph/leaf/constant.hpp" 

#ifndef TENNCOR_FUNCTOR_HPP
#define TENNCOR_FUNCTOR_HPP

namespace nnet
{

//! backward transfer function, get gradient nodes; F: Nf -> Nb 
using BACKMAP_F = std::function<varptr(std::vector<std::pair<inode*,inode*> >)>; 

//! calculate output shape from argument shapes 
using SHAPER_F = std::function<tensorshape(std::vector<tensorshape>)>; 
 
using USHAPE_F = std::function<tensorshape(tensorshape)>; 

using TENSOP_F = std::function<tensor*(std::unique_ptr<idata_src>&,std::vector<inode*>)>;

using DERIVE_F = std::function<varptr(inode*,std::vector<inode*>)>;

class functor final : public inode, public iobserver
{
public:
	static functor* get (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, std::string label);

	virtual ~functor (void);

	//! clone function
	functor* clone (void) const;

	//! move function
	functor* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual functor& operator = (const functor& other);

	//! declare move assignment to move over transfer functions
	virtual functor& operator = (functor&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get unique label with arguments
	virtual std::string get_name (void) const;

	// >>>>>> SERIALIZTAION DATA <<<<<<

	virtual NODE_TYPE node_type (void) const
	{
		return FUNCTOR_T;
	}

	// >>>>>> CONNECTION QUERY <<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<const inode*> get_leaves (void) const;

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! get all observerables
	std::vector<inode*> get_arguments (void) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

private:
	functor (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, std::string label);

	//! declare copy constructor to copy over transfer functions
	functor (const functor& other);

	//! declare move constructor to move over transfer functions
	functor (functor&& other);

	inode* clone_impl (void) const;

	inode* move_impl (void);

	//! copy helper
	void copy_helper (const functor& other);

	//! move helper
	void move_helper (functor&& other);



	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	virtual void update (void);

	// >>>>>>>>>>>> KILL CONDITION <<<<<<<<<<<<

	//! suicides when any dependency dies
	virtual void death_on_broken (void);


	TENSOP_F tensop_;
	
	DERIVE_F derive_;

	std::unique_ptr<idata_src> io_ = nullptr;

	// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	std::unique_ptr<tensor> data_ = nullptr;
};

}

#endif /* TENNCOR_FUNCTOR_HPP */
