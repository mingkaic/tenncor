#include "soil/typer.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"

#ifdef TYPER_HPP

DTYPE reg_typer (std::vector<DTYPE> args)
{
	if (0 == args.size())
	{
		handle_error("typing with no arguments");
	}
	DTYPE outtype = args.front();
	for (auto it = args.begin() + 1, et = args.end();
		it != et; ++it)
	{
		if (outtype != *it)
		{
			handle_error("incompatible types",
				ErrArg<std::string>{"first_type", name_type(outtype)},
				ErrArg<std::string>{"cur_type", name_type(*it)});
		}
	}
	return outtype;
}

static EnumMap<OPCODE,Typer> typers =
{
	{TRANSPOSE, reg_typer},
	{ADD, reg_typer},
	{MUL, reg_typer},
	{MATMUL, reg_typer},
};

Typer get_typer (OPCODE opcode)
{
	auto it = typers.find(opcode);
	if (typers.end() == it)
	{
		handle_error("failed to retrieve typer",
			ErrArg<std::string>("opcode", opname(opcode)));
	}
	return it->second;
}

#endif
