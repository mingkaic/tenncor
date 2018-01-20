//
//  mutable_connector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/iconnector.hpp"

#ifdef false

#pragma once
#ifndef mutable_connect_hpp
#define mutable_connect_hpp

#include <functional>

namespace nnet
{

using MAKE_CONNECT = std::function<inode*(std::vector<varptr>&)>;

// designed to cover all the edge cases of mutable connectors
// created permanent connector ic_ can potentially destroy mutable connector
// if arguments are destroyed (triggering a chain reaction)

class mutable_connector : public iconnector
{
	private:
		// we don't listen to inode when it's incomplete
		MAKE_CONNECT op_maker_;
		std::vector<varptr> arg_buffers_;
		// ic_ is a potential dependency
		iconnector* ic_ = nullptr;

		void connect (void);
		void disconnect (void);

	protected:
		mutable_connector (MAKE_CONNECT maker, size_t nargs);
		// ic_ uniqueness forces explicit copy constructor
		mutable_connector (const mutable_connector& other);

	public:
		static mutable_connector* get (MAKE_CONNECT maker, size_t nargs);

		virtual ~mutable_connector (void);

		// COPY
		virtual mutable_connector* clone (void);
		mutable_connector& operator = (const mutable_connector& other);

		// inode METHODS
		virtual tensorshape get_shape(void);

		virtual tensor<double>* get_eval(void);
	
		virtual varptr derive(void);

		// ICONNECTOR METHODS
		virtual void update (std::unordered_set<size_t> argidx);

		// MUTABLE METHODS
		// return true if replacing
		// replacing will destroy then remake ic_
		bool add_arg (inode* var, size_t idx);
		// return true if removing existing var at index idx
		bool remove_arg (size_t idx);
		bool valid_args (void);

		// ACCESSORS
		// get arguments
		size_t nargs (void) const;
		virtual void get_args (std::vector<inode*>& args) const;
};

}

#endif /* mutable_connect_hpp */

#endif
