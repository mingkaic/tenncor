#include "teq/ilayer.hpp"

#include "eteq/etens.hpp"

#ifndef ETEQ_LAYER_HPP
#define ETEQ_LAYER_HPP

namespace eteq
{

template <typename T>
struct Layer final : public teq::iLayer
{
	static Layer<T>* get (teq::Opcode opcode,
		teq::TensptrsT inputs, teq::FuncptrT output)
	{
		if (inputs.empty())
		{
			logs::fatalf("cannot perform `%s` without arguments",
				opcode.name_.c_str());
		}

		egen::_GENERATED_DTYPE tcode = egen::get_type<T>();
		for (teq::TensptrT child : inputs)
		{
			if (tcode != child->type_code())
			{
				logs::fatalf("incompatible tensor types %s and %s: "
					"cross-type functors not supported yet",
					egen::name_type(tcode).c_str(),
					child->type_label().c_str());
			}
		}
		return new Layer<T>(opcode, inputs, output);
	}

	/// Return deep copy of this FuncSignature
	Layer<T>* clone (void) const
	{
		return static_cast<Layer<T>*>(clone_impl());
	}

	Layer& operator = (const Layer& other) = delete;

	Layer (Layer&& other) = delete;

	Layer& operator = (Layer&& other) = delete;

	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_key) const override
	{
		return nullptr; // todo: implement
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return {}; // todo: implement
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		// todo: implement
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
	{
		// todo: implement
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_children (void) const override
	{
		return children_;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index >= children_.size())
		{
			logs::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, children_.size());
		}
		if (arg != children_[index])
		{
			children_[index] = arg;
		}
	}

	/// Implementation of iFunctor
	void calc (void) override
	{
		output_->calc();
	}

	/// Implementation of iData
	void* data (void) override
	{
		return output_->data();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return output_->data();
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return output_->nbytes();
	}

	/// Implementation of iLayer
	teq::TensptrT get_root (void) const override
	{
		return output_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return teq::TensptrsT(storages_.begin(), storages_.end());
	}

private:
	Layer (teq::Opcode opcode, teq::TensptrsT inputs, teq::FuncptrT output) :
        opcode_(opcode), children_(inputs), output_(output)
	{
		populate_storage();
	}

	Layer (const Layer& other) : opcode_(other.opcode_)
	{
		teq::iTensor* ooutput = other.output_.get();
		std::vector<teq::iTensor*> oinputs;
		auto& ochildren = other.children_;
		oinputs.reserve(ochildren.size());
		std::transform(ochildren.begin(), ochildren.end(),
			std::back_inserter(oinputs),
			[](teq::TensptrT ochild)
			{
				return ochild.get();
			});
		teq::Copier kamino(teq::TensSetT(oinputs.begin(), oinputs.end()));
		ooutput->accept(kamino);

		output_ = std::static_pointer_cast<teq::iFunctor>(kamino.clones_[ooutput]);
		children_.clear();
		for (auto oinput : oinputs)
		{
			children_.push_back(kamino.clones_[oinput]);
		}
		populate_storage();
	}

	iTensor* clone_impl (void) const override
	{
		return new Layer(*this);
	}

	void populate_storage (void)
	{
		// find all variables between children output_ and children_
		teq::GraphStat stats;
		for (auto child : children_)
		{
			stats.graphsize_.emplace(child.get(), estd::NumRange<size_t>());
		}
		output_->accept(stats);
		teq::OwnerMapT owner = teq::track_owners({output_});

		teq::TensSetT inputs;
		std::transform(children_.begin(), children_.end(),
			std::inserter(inputs, inputs.end()),
			[](teq::TensptrT child)
			{
				return child.get();
			});
		for (auto gpair : stats.graphsize_)
		{
			if (0 == gpair.second.upper_ &&
				false == estd::has(inputs, gpair.first))
			{
				storages_.push_back(std::static_pointer_cast<
					Variable<T>>(owner.at(gpair.first).lock()));
			}
		}
	}

    teq::Opcode opcode_;

	teq::TensptrsT children_;

	teq::FuncptrT output_;

	VarptrsT<T> storages_;
};

/// Smart pointer of layer
template <typename T>
using LayerptrT = std::shared_ptr<Layer<T>>;

}

#endif // ETEQ_LAYER_HPP
