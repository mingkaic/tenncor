#include "teq/ilayer.hpp"

#include "eteq/link.hpp"

#ifndef ETEQ_LAYER_HPP
#define ETEQ_LAYER_HPP

namespace eteq
{

template <typename T>
struct Layer final : public teq::iLayer
{
	Layer (teq::Opcode opcode, LinkptrT<T> input,
		LinkptrT<T> output, teq::TensptrsT storages) :
        opcode_(opcode), input_(input), output_(output), storages_(storages) {}

	Layer (const Layer& other) : opcode_(other.opcode_)
	{
		teq::iTensor* oinput = other.input_->get_tensor().get();
		teq::iTensor* ooutput = other.output_->get_tensor().get();
		teq::Copier kamino({oinput});
		ooutput->accept(kamino);

		input_ = to_link<T>(kamino.clones_[oinput]);
		output_ = to_link<T>(kamino.clones_[ooutput]);

        for (auto storage : other.storages_)
        {
            storages_.push_back(estd::try_get(
                kamino.clones_, storage.get(), nullptr));
        }
	}

	Layer& operator = (const Layer& other) = delete;

	Layer (Layer&& other) = delete;

	Layer& operator = (Layer&& other) = delete;

	/// Return deep copy of this FuncSignature
	Layer<T>* clone (void) const
	{
		return static_cast<Layer<T>*>(clone_impl());
	}

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
		return {input_->get_tensor()};
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		input_ = to_link<T>(arg);
	}

	/// Implementation of iSignature
	bool can_build (void) const override
	{
		return output_->can_build();
	}

	/// Implementation of iSignature
	teq::DataptrT build_data (void) const override
	{
		return output_->build_data();
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		return output_->shape_sign();
	}

	/// Implementation of iLayer
	teq::TensptrT get_root (void) const override
	{
		return output_->get_tensor();
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return storages_;
	}

private:
	iTensor* clone_impl (void) const override
	{
		return new Layer(*this);
	}

    teq::Opcode opcode_;

	LinkptrT<T> input_;

	LinkptrT<T> output_;

	teq::TensptrsT storages_;
};

/// Smart pointer of layer
template <typename T>
using LayerptrT = std::shared_ptr<Layer<T>>;

template <typename T>
struct LayerLink final : public iLink<T>
{
	LayerLink (LayerptrT<T> layer) : layer_(layer)
	{
		if (layer == nullptr)
		{
			logs::fatal("cannot link a null layer");
		}
	}

	/// Return deep copy of this instance (with a copied layer)
	LayerLink<T>* clone (void) const
	{
		return static_cast<LayerLink<T>*>(clone_impl());
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return layer_->ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_key) const override
	{
		return layer_->get_attr(attr_key);
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		layer_->add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
	{
		layer_->rm_attr(attr_key);
	}

	/// Implementation of iLink<T>
	teq::TensptrT get_tensor (void) const override
	{
		return layer_;
	}

	/// Implementation of iSignature
	bool can_build (void) const override
	{
		return layer_->can_build();
	}

	/// Implementation of iSignature
	teq::DataptrT build_data (void) const override
	{
		return layer_->build_data();
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		teq::Shape shape = layer_->shape();
		return teq::ShapeSignature(
			std::vector<teq::DimT>(shape.begin(), shape.end()));
	}

private:
	LayerLink (const LayerLink<T>& other) = default;

	iLink<T>* clone_impl (void) const override
	{
		return new LayerLink(LayerptrT<T>(layer_->clone()));
	}

	/// Implementation of iLink<T>
	void subscribe (Functor<T>* parent) override {}

	/// Implementation of iLink<T>
	void unsubscribe (Functor<T>* parent) override {}

	LayerptrT<T> layer_;
};

}

#endif // ETEQ_LAYER_HPP
