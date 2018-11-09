#include <string>
#include <vector>
#include <ostream>

#ifndef AGE_FILES_HPP
#define AGE_FILES_HPP

using StringsT = std::vector<std::string>;

static const char* macros = "_GENERATED_API_HPP";

struct Func final
{
	void defn (std::ostream& out) const
	{
		out << signature_ << "\n";
		out << "{\n";
		for (std::string line : lines_)
		{
			out << "\t" << line << ";\n";
		}
		if (retval_.size() > 0)
		{
			out << "\treturn " << retval_ << "\n";
		}
		out << "}";
	}

	std::string signature_;

	std::string retval_;

	StringsT lines_;
};

struct Enum final
{
	std::string name_;

	StringsT nums_;
}

struct File final
{
	void decl (std::ostream& header) const
	{
		for (std::string include : includes_)
		{
			header << "#include " << include << "\n";
		}
		header << "\n";
		header << "#ifndef " << macros << "\n";
		header << "#define " << macros << "\n\n";
		for (const Func& func : funcs_)
		{
			header << func.signature_ << ";\n\n";
		}
		header << "#endif\n";
	}

	void defn (std::ostream& source) const
	{
		source << "#ifdef " << macros << "\n\n";
		for (const Func& func : funcs_)
		{
			func.defn(source);
			source << "\n\n";
		}
		source << "#endif\n";
	}

	StringsT includes_;

	std::vector<Enum> enums_;

	std::vector<Func> funcs_;
};

#endif // AGE_FILES_HPP
