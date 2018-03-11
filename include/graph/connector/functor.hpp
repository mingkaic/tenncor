#include "include/graph/connector/iconnector.hpp"

#ifndef TENNCOR_FUNCTOR_HPP
#define TENNCOR_FUNCTOR_HPP

namespace nnet
{

using TENSOP_F = std::function<tensor*(std::unique_ptr<idata_src>&,std::vector<inode*>)>;

using DERIVE_F = std::function<varptr(inode*,std::vector<inode*>)>;

class functor final : public inode, public iobserver
{
public:
	static functor* get (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, std::string label)
	{
		assert(false == args.empty());
		return new functor(args, tensop, derive, label);
	}

	virtual ~functor (void) {}

	//! clone function
	functor* clone (void) const
	{
		return static_cast<functor*>(this->clone_impl());
	}

	//! move function
	functor* move (void)
	{
		return static_cast<functor*>(this->move_impl());
	}

	//! declare copy assignment to copy over transfer functions
	virtual functor& operator = (const functor& other)
	{
		if (this != &other)
		{
			iobserver::operator = (other);
			inode::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	//! declare move assignment to move over transfer functions
	virtual functor& operator = (functor&& other)
	{
		if (this != &other)
		{
			iobserver::operator = (std::move(other));
			inode::operator = (std::move(other));
			move_helper(std::move(other));
		}
		return *this;
	}



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get unique label with arguments
	virtual std::string get_name (void) const
	{
		std::string args;
		auto it = this->dependencies_.begin();
		auto et = this->dependencies_.end();
		const inode * arg = dynamic_cast<const inode*>(*it);
		while (args.empty() && nullptr == arg)
		{
			arg = dynamic_cast<const inode*>(*++it);
		}
		if (arg)
		{
			args = arg->get_label();
			++it;
		}
		while (it != et)
		{
			if (nullptr != (arg = dynamic_cast<const inode*>(*it)))
			{
				args += "," + arg->get_label();
			}
			it++;
		}
		return inode::get_name() + "(" + args + ")";
	}

	// >>>>>> CONNECTION QUERY <<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<inode*> get_leaves (void) const
	{
		std::unordered_set<inode*> leaves;
		std::vector<inode*> args = this->get_arguments();
		for (inode* arg : args)
		{
			std::unordered_set<inode*> subleaves = arg->get_leaves();
			leaves.insert(subleaves.begin(), subleaves.end());
		}
		return leaves;
	}

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! get all observerables
	std::vector<inode*> get_arguments (void) const
	{
		std::vector<inode*> node_args(this->dependencies_.size());
		std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
			[](subject* s) { return static_cast<inode*>(s); });
		return node_args;
	}



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void)
	{
		return data_.get();
	}

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt)
	{
		if (wrt == this)
		{
			tensor* ten = wrt->get_tensor();
			assert(ten && ten->has_data());
			tensorshape shape = ten->get_shape();
			std::vector<double> data(shape.n_elems(), 1); // change to match wrt type
			return constant::get(data, shape);
		}
		if (nullptr == data_)
		{
			throw std::exception(); // uninitialized variables
		}
		return derive_(wrt, get_arguments());
	}

	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	virtual void update (void)
	{
		std::vector<inode*> args = get_arguments();
		bool has_data = std::all_of(args.begin(), args.end(), 
		[](inode* node)
		{
			tensor* tens = node->get_tensor();
			return nullptr != tens && tens->has_data();
		});
		if (has_data)
		{
			if (nullptr == data_)
			{
				data_ = std::unique_ptr<tensor>(tensop_(io_, args));
			}
			data_->read_from(*io_);
			this->notify(UPDATE);
		}
	}

private:
	functor (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, std::string label) :
		inode(label), iobserver(std::vector<subject*>(args.begin(), args.end())),
		tensop_(tensop), derive_(derive)
	{ this->update(); }

	//! declare copy constructor to copy over transfer functions
	functor (const functor& other) : 
		inode(other), iobserver(other)
	{ copy_helper(other); }

	//! declare move constructor to move over transfer functions
	functor (functor&& other) : 
		inode(std::move(other)), iobserver(std::move(other))
	{ move_helper(std::move(other)); }

	inode* clone_impl (void) const
	{
		return new functor(*this);
	}

	inode* move_impl (void)
	{
		return new functor(std::move(*this));
	}

	//! copy helper
	void copy_helper (const functor& other)
	{
		tensop_ = other.tensop_;
		derive_ = other.derive_;
		if (nullptr != other.io_)
		{
			io_ = std::unique_ptr<idata_src>(other.io_->clone());
		}
		if (nullptr != other.data_)
		{
			data_ = std::make_unique<tensor>(*other.data_);
		}
	}

	//! move helper
	void move_helper (functor&& other)
	{
		tensop_ = std::move(other.tensop_);
		derive_ = std::move(other.derive_);
		io_ = std::move(other.io_);
		data_ = std::move(other.data_);
	}

	// >>>>>>>>>>>> KILL CONDITION <<<<<<<<<<<<

	//! suicides when any dependency dies
	virtual void death_on_broken (void)
	{
		delete this;
	}


	TENSOP_F tensop_;
	
	DERIVE_F derive_;

	std::unique_ptr<idata_src> io_ = nullptr;

	// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	std::unique_ptr<tensor> data_ = nullptr;
};

}

#endif /* TENNCOR_FUNCTOR_HPP */
