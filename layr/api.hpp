///
/// dense.hpp
/// layr
///
/// Purpose:
/// Implement fully connected layer
///

#include "eteq/generated/api.hpp"
#include "eteq/serialize.hpp"

#include "layr/init.hpp"

#ifndef LAYR_API_HPP
#define LAYR_API_HPP

namespace layr
{

/// Fully connected weight label
const std::string weight_key = "weight";

/// Fully connected bias label
const std::string bias_key = "bias";

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

template <typename T>
using UnaryF = std::function<eteq::ETensor<T>(eteq::ETensor<T>)>;

static inline teq::TensMapT<std::string> replace_targets (
	const teq::TensMapT<teq::TensptrT>& inputs)
{
	teq::TensMapT<std::string> targets;
	for (auto& inp : inputs)
	{
		targets.emplace(inp.first, "target");
	}
	return targets;
}

template <typename T>
struct Trailer final : public teq::OnceTraveler
{
	Trailer (const teq::TensMapT<teq::TensptrT>& inputs) :
		trailed_(inputs), pfinder_(replace_targets(inputs)) {}

	teq::TensMapT<teq::TensptrT> trailed_;

private:
	/// Implementation of OnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override {}

	/// Implementation of OnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		if (estd::has(trailed_, &func))
		{
			return;
		}
		func.accept(pfinder_);
		if (false == estd::has(pfinder_.roadmap_, &func))
		{
			return;
		}
		auto& target_dir = pfinder_.roadmap_.at(&func).at("target");

		marsh::Maps dup_attrs;
		marsh::get_attrs(dup_attrs, func);
		auto& attr_directions = target_dir.attrs_;
		for (std::string attr : attr_directions)
		{
			auto ref = static_cast<const teq::TensorRef*>(func.get_attr(attr));
			auto ctens = ref->get_tensor();
			ctens->accept(*this);
			dup_attrs.rm_attr(attr);
			dup_attrs.add_attr(attr, marsh::ObjptrT(ref->copynreplace(
				trailed_.at(ctens.get()))));
		}

		auto& child_directions = target_dir.children_;
		teq::TensptrsT children = func.get_children();
		for (size_t i : child_directions)
		{
			auto child = children[i];
			child->accept(*this);
			children[i] = trailed_.at(child.get());
		}

		trailed_.emplace(&func, eteq::Functor<T>::get(
			(egen::_GENERATED_OPCODE) func.get_opcode().code_,
			children, std::move(dup_attrs)));
	}

	teq::PathFinder pfinder_;
};

/// Copy everything from input.first to root, except replacing input.first with input.second
template <typename T>
eteq::ETensor<T> trail (const eteq::ETensor<T>& root,
	const teq::TensMapT<teq::TensptrT>& inputs)
{
	Trailer<T> trailer(inputs);
	root->accept(trailer);
	return estd::try_get(trailer.trailed_, root.get(), nullptr);
}

template <typename T>
eteq::VarptrsT<T> calc_storage (teq::TensptrT root, teq::TensptrT input)
{
	// find all variables between children output_ and children_
	teq::GraphStat stats;
	stats.graphsize_.emplace(input.get(), estd::NumRange<size_t>());
	root->accept(stats);
	teq::OwnerMapT owner = teq::track_owners({root});

	eteq::VarptrsT<T> storages;
	for (auto gpair : stats.graphsize_)
	{
		if (0 == gpair.second.upper_ && input.get() != gpair.first)
		{
			if (auto var = std::dynamic_pointer_cast<
				eteq::Variable<T>>(owner.at(gpair.first).lock()))
			{
				storages.push_back(var);
			}
		}
	}
	return storages;
}

template <typename T>
void get_storage (teq::TensptrsT& out, const eteq::ETensor<T>& root)
{
	if (auto f = dynamic_cast<const teq::iFunctor*>(root.get()))
	{
		if (auto lattr = f->get_attr(teq::layer_key))
		{
			auto layer = static_cast<const teq::LayerObj*>(lattr);
			auto storage = calc_storage<T>(root, layer->get_tensor());
			out.insert(out.end(), storage.begin(), storage.end());
		}
		auto children = f->get_children();
		for (auto child : children)
		{
			get_storage(out, eteq::ETensor<T>(child));
		}
	}
}

template <typename T>
UnaryF<T> dense_builder (teq::Shape inshape,
	std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	teq::TensptrT weight = weight_init(gen_rshape(
		hidden_dims, inshape, dims), weight_key);
	teq::TensptrT bias;
	if (bias_init)
	{
		bias = bias_init(teq::Shape(hidden_dims), bias_key);
	}
	return [weight, bias, dims](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input,
				eteq::ETensor<T>(weight), eteq::ETensor<T>(bias), dims);
		};
}

template <typename T>
UnaryF<T> conv_builder (std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	teq::TensptrT weight = unif_xavier_init<T>(1)(teq::Shape({out_ncol,
		in_ncol, filter_hw.second, filter_hw.first}), weight_key);
	teq::TensptrT bias = zero_init<T>()(teq::Shape({out_ncol}), bias_key);
	return [weight, bias, zero_padding](eteq::ETensor<T> input)
		{
			return tenncor::layer::conv(input, eteq::ETensor<T>(weight),
				eteq::ETensor<T>(bias), zero_padding);
		};
}

template <typename T>
UnaryF<T> rnn_builder (teq::DimT indim, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	auto cell = dense_builder(
		teq::Shape({hidden_dim, (teq::DimT) (hidden_dim + indim)}),
		teq::Shape({hidden_dim}), weight_init, bias_init);
	auto init_state = eteq::make_variable<T>(
		teq::Shape({hidden_dim}), "init_state");
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
		teq::DimT nseq = inshape.at(seq_dim);
		if (nseq == 0)
		{
			logs::fatalf("cannot sequence on ambiguous dimension %d on shape %s",
				seq_dim, inshape.to_string().c_str());
		}
		if (seq_dim == 0)
		{
			logs::fatalf("spliting input across 0th dimension... "
				"dense connection will not match");
		}
		eteq::ETensorsT<T> states;
		auto inslice = tenncor::slice(input, 0, 1, seq_dim);
		auto state = activation(cell(tenncor::concat(inslice,
			tenncor::extend_like(eteq::ETensor<T>(init_state), inslice), 0)));
		states.push_back(state);
		for (teq::DimT i = 1; i < nseq; ++i)
		{
			inslice = tenncor::slice(input, i, 1, seq_dim);
			state = activation(cell(tenncor::concat(inslice, state, 0)));
			states.push_back(state);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_RNN_LAYER", 0}, {input}, output);
	};
}

template <typename T>
UnaryF<T> lstm_builder (teq::DimT indim, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + indim)});
	teq::Shape bshape({hidden_dim});
	auto ggate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto forgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto ingate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto outgate = dense_builder(wshape, bshape, weight_init, bias_init);
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
		teq::DimT nseq = inshape.at(seq_dim);
		if (nseq == 0)
		{
			logs::fatalf("cannot sequence on ambiguous dimension %d on shape %s",
				seq_dim, inshape.to_string().c_str());
		}
		if (seq_dim == 0)
		{
			logs::fatalf("spliting input across 0th dimension... "
				"dense connection will not match");
		}
		teq::Shape stateshape({hidden_dim});
		auto prevstate = eteq::make_constant_scalar<T>(0, stateshape);
		auto prevhidden = eteq::make_constant_scalar<T>(0, stateshape);
		eteq::ETensorsT<T> states;
		for (teq::DimT i = 0; i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			eteq::ETensor<T> xc = tenncor::concat(inslice, prevhidden, 0);

			auto gate = tenncor::tanh(ggate(xc));
			auto input = tenncor::sigmoid(ingate(xc));
			auto forget = tenncor::sigmoid(forgate(xc));
			auto output = tenncor::sigmoid(outgate(xc));
			prevstate = gate * input + prevstate * forget;
			prevhidden = prevstate * output;
			states.push_back(prevhidden);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_LSTM_LAYER", 0}, input, output);
	};
}

template <typename T>
UnaryF<T> gru_builder (teq::DimT indim, eteq::ETensor<T> input,
	teq::DimT hidden_dim, UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + indim)});
	teq::Shape bshape({hidden_dim});
	auto ugate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto rgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto hgate = dense_builder(wshape, bshape, weight_init, bias_init);
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
		teq::DimT nseq = inshape.at(seq_dim);
		if (nseq == 0)
		{
			logs::fatalf("cannot sequence on ambiguous dimension %d "
				"on shape %s", seq_dim, inshape.to_string().c_str());
		}
		if (seq_dim == 0)
		{
			logs::fatalf("spliting input across 0th dimension... "
				"dense connection will not match");
		}
		teq::Shape stateshape({hidden_dim});
		auto state = eteq::make_constant_scalar<T>(0, stateshape);
		eteq::ETensorsT<T> states;
		for (teq::DimT i = 0; i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			eteq::ETensor<T> xc = tenncor::concat(inslice, state, 0);
			auto update = tenncor::sigmoid(ugate(xc));
			auto reset = tenncor::sigmoid(rgate(xc));
			auto hidden = tenncor::tanh(hgate(
				tenncor::concat(inslice, reset * state, 0)));
			state = update * state + ((T) 1 - update) * hidden;
			states.push_back(state);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_GRU_LAYER", 0}, input, output);
	};
}

template <typename T>
struct RBMBuilder final
{
	UnaryF<T> fwd_;

	UnaryF<T> bwd_;
};

/// Returns forward builder, and assigns backward builder
template <typename T>
RBMBuilder<T> rbm_builder (
	teq::DimT nhidden, teq::DimT nvisible,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init)
{
	teq::TensptrT weight = weight_init(
		teq::Shape({nhidden, nvisible}), weight_key);
	teq::TensptrT hbias;
	teq::TensptrT vbias;
	if (bias_init)
	{
		hbias = bias_init(teq::Shape({nhidden}), "h" + bias_key);
		vbias = bias_init(teq::Shape({nvisible}), "v" + bias_key);
	}
	return RBMBuilder<T>{
		[weight, hbias](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input, eteq::ETensor<T>(weight),
				eteq::ETensor<T>(hbias), {{0, 1}});
		},
		[weight, vbias](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input, tenncor::transpose(eteq::ETensor<T>(weight)),
				eteq::ETensor<T>(vbias), {{0, 1}});
		}
	};
}

template <typename T>
eteq::ETensor<T> dense (eteq::ETensor<T> input,
	std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	return dense_builder<T>(input->shape(), hidden_dims,
		weight_init, bias_init, dims)(input);
}

template <typename T>
eteq::ETensor<T> conv (eteq::ETensor<T> input,
	std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	return conv_builder<T>(filter_hw, in_ncol, out_ncol, zero_padding)(input);
}

template <typename T>
eteq::ETensor<T> rnn (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return rnn_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

template <typename T>
eteq::ETensor<T> lstm (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return lstm_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

template <typename T>
eteq::ETensor<T> gru (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return gru_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

const std::string input_key = "model_input";

template <typename T>
struct ManagedModel final
{
	ManagedModel (void) = default;

	ManagedModel (eteq::ETensor<T> root, eteq::ETensor<T> input) :
		root_(root), input_(input) {}

	ManagedModel (const ManagedModel& other)
	{
		copy_helper(other);
	}

	ManagedModel& operator = (const ManagedModel& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	ManagedModel (ManagedModel&& other) = default;

	ManagedModel& operator = (ManagedModel&& other) = default;

	UnaryF<T> get_builder (void)
	{
		return [this](eteq::ETensor<T> oinput)
		{
			return this->retrail(oinput);
		};
	}

	/// Trail root to input, replacing input with oinput
	eteq::ETensor<T> retrail (eteq::ETensor<T> oinput) const
	{
		return trail(root_, teq::TensMapT<teq::TensptrT>{{input_.get(), oinput}});
	}

	void save (onnx::ModelProto& model) const
	{
		onnx::TensIdT ids;
		ids.insert({input_.get(), input_key});
		eteq::save_model(model, {root_}, ids);
	}

	void load (const onnx::ModelProto& model)
	{
		onnx::TensptrIdT ids;
		auto roots = eteq::load_model(ids, model);
		if (roots.empty())
		{
			logs::fatal("cannot load model without roots");
		}
		input_ = estd::must_getf(ids.right, input_key,
			"failed to find %s", input_key.c_str());
		root_ = roots.front();
	}

private:
	void copy_helper (const ManagedModel& other)
	{
		input_ = other.input_;
		teq::Copier kamino({input_.get()});
		auto oroot = other.root_;
		oroot->accept(kamino);
		root_ = kamino.clones_.at(oroot.get());
	}

	eteq::ETensor<T> root_;

	eteq::ETensor<T> input_;
};

}

#endif // LAYR_API_HPP
