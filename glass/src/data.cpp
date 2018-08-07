#include "glass/data.hpp"

#include "soil/variable.hpp"

#ifdef GLASS_DATA_HPP

void save_data (tenncor::DataRepoPb& out, const Session& in)
{
	out.set_id(in.hash());
	std::list<iNode*> nodes = order_nodes(in);
	auto it = nodes.begin();
	for (size_t i = 0, n = nodes.size(); i < n; ++i)
	{
		if (dynamic_cast<Variable*>(*it))
		{
			tenncor::DataPb varpb;
			save_node(varpb, *it);
			out.mutable_data_map()->insert({(uint32_t) i, varpb});
		}
		++it;
	}
}

#define SET_VAR(TYPE)\
TYPE* dptr = (TYPE*) ptr;\
var->set_data(std::vector<TYPE>(dptr, dptr + n));

void load_data (Session& out, const tenncor::DataRepoPb& data)
{
	std::string sessid = out.hash();
	std::string dataid = data.id();
	if (0 != sessid.compare(dataid))
	{
		handle_error("incompatible graph and session id",
			ErrArg<std::string>("sessid", sessid),
			ErrArg<std::string>("dataid", dataid));
	}
	std::list<iNode*> nodes = order_nodes(out);
	std::vector<iNode*> nodevec(nodes.begin(), nodes.end());
	auto datamap = data.data_map();
	for (auto datapair : datamap)
	{
		if (Variable* var = dynamic_cast<Variable*>(nodevec[datapair.first]))
		{
			std::string data;
			Shape shape;
			DTYPE type = load_node(data, shape, datapair.second);
			char* ptr = &data[0];

			Shape varshape = var->shape();
			if (false == varshape.compatible_after(shape, 0))
			{
				handle_error("fail to set data of incompatible shapes",
					ErrArg<std::string>("varshape", varshape.to_string()),
					ErrArg<std::string>("pbshape", shape.to_string())); // todo: make warn instead
			}

			DTYPE vartype = var->type();
			if (vartype != type)
			{
				handle_error("fail to set data of incompatible types",
					ErrArg<std::string>("vartype", name_type(vartype)),
					ErrArg<std::string>("pbshape", name_type(type))); // todo: make warn instead
			}

			NElemT n = shape.n_elems();
			switch (type)
			{
				case DOUBLE:
				{
					SET_VAR(double)
				}
				break;
				case FLOAT:
				{
					SET_VAR(float)
				}
				break;
				case INT8:
				{
					SET_VAR(int8_t)
				}
				break;
				case INT16:
				{
					SET_VAR(int16_t)
				}
				break;
				case INT32:
				{
					SET_VAR(int32_t)
				}
				break;
				case INT64:
				{
					SET_VAR(int64_t)
				}
				break;
				case UINT8:
				{
					SET_VAR(uint8_t)
				}
				break;
				case UINT16:
				{
					SET_VAR(uint16_t)
				}
				break;
				case UINT32:
				{
					SET_VAR(uint32_t)
				}
				break;
				case UINT64:
				{
					SET_VAR(uint64_t)
				}
				break;
				default:
					handle_error("failed to deserialize badly typed node"); // todo: make warn instead
			}
		}
		else
		{
			handle_error("failed to set data at incorrect index",
				ErrArg<uint32_t>("index", datapair.first)); // todo: make warn instead
		}
	}
}

#endif
