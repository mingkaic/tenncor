#include <string>
#include <vector>
#include <ostream>
#include <iomanip>

#ifndef AGE_FILES_HPP
#define AGE_FILES_HPP

using StringsT = std::vector<std::string>;

struct Stmt
{
	virtual ~Stmt (void) = default;

	virtual void defn (std::ostream& out, std::string pfx = "") const = 0;
};

using StmtptrT = std::shared_ptr<Stmt>;

using StmtsT = std::vector<StmtptrT>;

struct VarDecl final : public Stmt
{
	VarDecl (std::string type, std::string name) : type_(type), name_(name) {}

	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << type_ << " " << name_ << ";";
	}

	std::string type_;

	std::string name_;
};

struct VarAssign final : public Stmt
{
	VarAssign (std::string name, std::string val) : name_(name), val_(val) {}

	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << name_ << " = " << val_ << ";";
	}

	std::string name_;

	std::string val_;
};

struct ReturnStmt final : public Stmt
{
	ReturnStmt (std::string val) : retval_(val) {}

	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << "return " << retval_ << ";";
	}

	std::string retval_;
};

struct SwitchStmt final : public Stmt
{
	SwitchStmt (std::string code) : code_(code) {}

	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << "switch (" << code_ << ")\n";
		out << pfx << "{\n";
		for (auto cpair : cases_)
		{
			out << pfx << "\t" << "case " << cpair.first << ":\n";
			out << pfx << "\t" << "{\n";
			cpair.second->defn(out, pfx + "\t\t");
			out << "\n";
			out << pfx << "\t" << "}\n";
		}
		out << pfx << "}";
	}

	std::string code_;

	std::unordered_map<std::string,StmtptrT> cases_;
};

struct MapRep final
{
	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << "std::unordered_map<" << types_.first << "," <<
			types_.second << "> " << name_ << " =\n";
		out << pfx << "{\n";
		for (auto& vpair : values_)
		{
			out << pfx << "\t{" << vpair.first << "," <<
				vpair.second << "},\n";
		}
		out << pfx << "};";
	}

	std::string name_;

	std::pair<std::string,std::string> types_;

	std::unordered_map<std::string,std::string> values_;

	bool is_static_;
};

struct Enum final
{
	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << "enum " << name_ << "\n";
		out << pfx << "{\n";
		if (nums_.size() > 0)
		{
			out << pfx << "\t" << nums_[0] << " = 0,\n";
			for (size_t i = 1, n = nums_.size(); i < n; ++i)
			{
				out << pfx << "\t"  << nums_[i] << ",\n";
			}
		}
		out << pfx << "};";
	}

	std::string name_;

	StringsT nums_;
};

struct Func final
{
	void defn (std::ostream& out, std::string pfx = "") const
	{
		out << pfx << signature_ << "\n";
		out << pfx << "{\n";
		for (const StmtptrT& stmt : lines_)
		{
			stmt->defn(out, pfx + "\t");
			out << "\n";
		}
		out << pfx << "}";
	}

	std::string signature_;

	StmtsT lines_;

	bool header_only_ = false;
};

struct File final
{
	File (std::string macro) : macro_(macro) {}

	File (std::string macro, StringsT includes) :
		macro_(macro), includes_(includes) {}

	void decl (std::ostream& header) const
	{
		for (std::string include : includes_)
		{
			header << "#include " << include << "\n";
		}
		header << "\n";
		header << "#ifndef " << macro_ << "\n";
		header << "#define " << macro_ << "\n\n";
		header << "namespace age\n";
		header << "{\n\n";
		for (const Enum& en : enums_)
		{
			en.defn(header, "");
			header << "\n\n";
		}
		for (const Func& func : funcs_)
		{
			if (func.header_only_)
			{
				func.defn(header, "");
				header << "\n\n";
			}
			else
			{
				header << func.signature_ << ";\n\n";
			}
		}
		header << "}\n\n";
		header << "#endif\n";
	}

	void defn (std::ostream& source) const
	{
		source << "#ifdef " << macro_ << "\n\n";
		source << "namespace age\n";
		source << "{\n\n";
		for (const MapRep& mrep : maps_)
		{
			source << "static ";
			mrep.defn(source, "");
			source << "\n\n";
		}
		for (const Func& func : funcs_)
		{
			if (false == func.header_only_)
			{
				func.defn(source, "");
				source << "\n\n";
			}
		}
		source << "}\n\n";
		source << "#endif\n";
	}

	std::string macro_;

	StringsT includes_;

	std::vector<MapRep> maps_;

	std::vector<Enum> enums_;

	std::vector<Func> funcs_;
};

#endif // AGE_FILES_HPP
