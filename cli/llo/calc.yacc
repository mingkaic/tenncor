%{
#include<stdio.h>

#include "cli/llo/ast/ast.h"

extern int yylex();
extern int yyerror();
extern FILE *yyin;

%}

%start list

%union
{
	int int_type;
	double num_type;
	char str_type[32];
	struct DataHolder* data_type;
	struct ShapeHolder* vec_type;
	struct ASTNode* ref_type;
}

%type <int_type> integer UNARY BINARY DIMOP SHAPEOP DIGIT
%type <num_type> number
%type <str_type> VAR
%type <data_type> arr
%type <vec_type> shape
%type <ref_type> expr

%token DIGIT VAR UNARY BINARY DIMOP SHAPEOP GRAD PRINT SHAPE EXIT MODE

%left '+' '-'
%left '*' '/'
%left UMINUS /*supplies precedence for unary minus */

%% /* beginning of rules section */

list: 	/* empty */
		|
		list EXIT
		{
			YYACCEPT;
		}
		|
		list stat '\n'
		|
		list error '\n'
		{
			yyerrok;
		}

stat:	/* empty */
		{}
		|
		MODE VAR
		{
			use_mode($2);
		}
		|
		expr
		{
			show_data($1);
			free_ast($1);
		}
		|
		VAR '=' expr
		{
			save_ast($1, $3);
			free_ast($3);
		}
		|
		PRINT
		{
			printf("\n");
		}
		|
		PRINT '(' expr ')'
		{
			show_eq($3);
			free_ast($3);
		}
		|
		PRINT '(' expr ',' SHAPE ')'
		{
			show_shape($3);
			free_ast($3);
		}
		;

expr:   GRAD '(' expr ',' VAR ')'
		{
			struct ASTNode* node = load_ast($5);
			$$ = grad($3, node);
			free_ast($3);
			free_ast(node);

		}
		|
		UNARY '(' expr ')'
		{
			$$ = unary($3, $1);
			free_ast($3);
		}
		|
		BINARY '(' expr ',' expr ')'
		{
			$$ = binary($3, $5, $1);
			free_ast($3);
			free_ast($5);
		}
		|
		DIMOP '(' expr ',' integer ')'
		{
			$$ = unary_dim($3, $5, $1);
			free_ast($3);
		}
		|
		SHAPEOP '(' expr ',' shape ')'
		{
			$$ = shapeop($3, $5, $1);
			free_ast($3);
			free_shape($5);
		}
		|
		'(' expr ')'
		{
			$$ = $2;
		}
		|
		expr '*' expr
		{
			$$ = binary($1, $3, MUL);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '/' expr
		{
			$$ = binary($1, $3, DIV);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '+' expr
		{
			$$ = binary($1, $3, ADD);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '-' expr
		{
			$$ = binary($1, $3, SUB);
			free_ast($1);
			free_ast($3);
		}
		|
		'-' expr %prec UMINUS
		{
			$$ = unary($2, NEG);
			free_ast($2);
		}
		|
		VAR
		{
			$$ = load_ast($1);
		}
		|
		'[' arr ']'
		{
			$$ = to_node($2);
			free_data($2);
		}
		|
		'[' ']'
		{
			$$ = empty_node();
		}
		;

arr:	arr ',' '[' arr ']'
		{
			data_append($1, $4);
			$$ = $1;
		}
		|
		'[' arr ']'
		{
			$$ = make_data($2);
		}
		|
		arr ',' number
		{
			data_append_d($1, $3);
			$$ = $1;
		}
		|
		number
		{
			$$ = make_data_d($1);
		}
		;

shape:	shape ',' integer
		{
			shape_append($1, $3);
			$$ = $1;
		}
		|
		integer
		{
			$$ = make_shape($1);
		}
		;

number: integer '.' integer
		{
			$$ = $1 + 1.0 / $3;
		}
		|
		integer
		{
			$$ = $1;
		};

integer: DIGIT
		{
		   $$ = $1;
		}
		|
		integer DIGIT
		{
		   $$ = 10 * $1 + $2;
		}
		;

%%
int main (int argc, char** argv)
{
	int exit_status = 0;
	if (argc > 1)
	{
		int i;
		for (i = 1; i < argc; ++i)
		{
			FILE* file = fopen(argv[i], "r");
			if (file)
			{
				yyin = file;
				exit_status &= yyparse();
				fclose(file);
			}
			else
			{
				// warn
			}
		}
	}
	else
	{
		exit_status = yyparse();
	}
	return exit_status;
}

int yyerror(s)
char *s;
{
	// todo: improve logger
	fprintf(stderr, "%s\n",s);
}

int yywrap()
{
	return(1);
}
