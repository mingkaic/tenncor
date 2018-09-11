%{
#include <stdio.h>
#include <math.h>

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

%type <int_type> INTEGER UNARY BINARY DIMOP SHAPEOP
%type <num_type> number DECIMAL
%type <str_type> VAR
%type <data_type> arr
%type <vec_type> shape
%type <ref_type> expr

%token INTEGER DECIMAL VAR UNARY BINARY DIMOP SHAPEOP GRAD PRINT SHAPE EXIT MODE
ASSIGN LPAREN RPAREN LSB RSB COMMA PLUS MINUS STAR SLASH NEWLINE

%left PLUS MINUS
%left STAR SLASH
%left UMINUS /*supplies precedence for unary minus */

%% /* beginning of rules section */

list: 	/* empty */
		|
		list EXIT
		{
			YYACCEPT;
		}
		|
		list stat NEWLINE
		|
		list error NEWLINE
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
		VAR ASSIGN expr
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
		PRINT LPAREN expr RPAREN
		{
			show_eq($3);
			free_ast($3);
		}
		|
		SHAPE LPAREN expr RPAREN
		{
			show_shape($3);
			free_ast($3);
		}
		;

expr:   GRAD LPAREN expr COMMA VAR RPAREN
		{
			struct ASTNode* node = load_ast($5);
			$$ = grad($3, node);
			free_ast($3);
			free_ast(node);
		}
		|
		UNARY LPAREN expr RPAREN
		{
			$$ = unary($3, $1);
			free_ast($3);
		}
		|
		BINARY LPAREN expr COMMA expr RPAREN
		{
			$$ = binary($3, $5, $1);
			free_ast($3);
			free_ast($5);
		}
		|
		DIMOP LPAREN expr COMMA INTEGER RPAREN
		{
			$$ = unary_dim($3, $5, $1);
			free_ast($3);
		}
		|
		SHAPEOP LPAREN expr COMMA shape RPAREN
		{
			$$ = shapeop($3, $5, $1);
			free_ast($3);
			free_shape($5);
		}
		|
		LPAREN expr RPAREN
		{
			$$ = $2;
		}
		|
		expr STAR expr
		{
			$$ = binary($1, $3, MUL);
			free_ast($1);
			free_ast($3);
		}
		|
		expr SLASH expr
		{
			$$ = binary($1, $3, DIV);
			free_ast($1);
			free_ast($3);
		}
		|
		expr PLUS expr
		{
			$$ = binary($1, $3, ADD);
			free_ast($1);
			free_ast($3);
		}
		|
		expr MINUS expr
		{
			$$ = binary($1, $3, SUB);
			free_ast($1);
			free_ast($3);
		}
		|
		MINUS expr %prec UMINUS
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
		LSB arr RSB
		{
			$$ = to_node($2);
			free_data($2);
		}
		|
		LSB RSB
		{
			$$ = empty_node();
		}
		;

arr:	arr COMMA LSB arr RSB
		{
			data_append($1, $4);
			$$ = $1;
		}
		|
		LSB arr RSB
		{
			$$ = make_data($2);
		}
		|
		arr COMMA number
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

shape:	shape COMMA INTEGER
		{
			shape_append($1, $3);
			$$ = $1;
		}
		|
		INTEGER
		{
			$$ = make_shape($1);
		}
		;

number:	DECIMAL
		{
			$$ = $1;
		}
		|
		INTEGER
		{
			$$ = $1;
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
