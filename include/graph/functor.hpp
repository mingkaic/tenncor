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
#include "include/graph/constant.hpp" 

#ifndef TENNCOR_FUNCTOR_HPP
#define TENNCOR_FUNCTOR_HPP

namespace nnet
{

#define OPCODE tenncor::opcode_t
#define _OP_SENTINEL tenncor::_OP_SENTINEL

#define ABS tenncor::ABS
#define NEG tenncor::NEG
#define NOT tenncor::NOT
#define SIN tenncor::SIN
#define COS tenncor::COS
#define TAN tenncor::TAN
#define EXP tenncor::EXP
#define LOG tenncor::LOG
#define SQRT tenncor::SQRT
#define ROUND tenncor::ROUND
#define POW tenncor::POW
#define ADD tenncor::ADD
#define SUB tenncor::SUB
#define MUL tenncor::MUL
#define DIV tenncor::DIV
#define EQ tenncor::EQ
#define NE tenncor::NE
#define GT tenncor::GT
#define LT tenncor::LT
#define BINO tenncor::BINO
#define UNIF tenncor::UNIF
#define NORM tenncor::NORM
#define TRANSPOSE tenncor::TRANSPOSE
#define FLIP tenncor::FLIP
#define ARGMAX tenncor::ARGMAX
#define RMAX tenncor::RMAX
#define RSUM tenncor::RSUM
#define EXPAND tenncor::EXPAND
#define N_ELEMS tenncor::N_ELEMS
#define N_DIMS tenncor::N_DIMS
#define MATMUL tenncor::MATMUL

#define INJACOBIAN tenncor::INJACOBIAN
#define OUTJACOBIAN tenncor::OUTJACOBIAN
#define JACOBIANLEFT tenncor::JACOBIANLEFT
#define JACOBIANRIGHT tenncor::JACOBIANRIGHT

//! backward transfer function, get gradient nodes; F: Nf -> Nb 
using BACKMAP_F = std::function<varptr(std::vector<std::pair<inode*,varptr> >)>; 

//! calculate output shape from argument shapes 
using SHAPER_F = std::function<tensorshape(std::vector<tensorshape>)>; 
 
using USHAPE_F = std::function<tensorshape(tensorshape, std::vector<uint64_t>)>; 

using TENSOP_F = std::function<tensor*(std::unique_ptr<idata_src>&,std::vector<inode*>)>;

using DERIVE_F = std::function<varptr(inode*,std::vector<inode*>)>;

class functor final : public inode, public iobserver
{
public:
	static functor* get (std::vector<inode*> args, TENSOP_F tensop, 
		DERIVE_F derive, OPCODE code);

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

	// >>>>>> CONNECTION QUERY <<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<const inode*> get_leaves (void) const;

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! get all observerables
	std::vector<inode*> get_arguments (void) const;

	OPCODE get_opcode (void) const
	{
		return opcode_;
	}



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

protected:
	// >>>>>> SERIALIZATION DATA <<<<<<

	virtual nnet::NODE_TYPE node_type (void) const;

	// >>>>>> SERIALIZATION ACTOR <<<<<<

	virtual void serialize_detail (google::protobuf::Any* proto_dest);
	
	friend class graph;

private:
	functor (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, OPCODE code);

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

	OPCODE opcode_ = _OP_SENTINEL;
};

}

#endif /* TENNCOR_FUNCTOR_HPP */
