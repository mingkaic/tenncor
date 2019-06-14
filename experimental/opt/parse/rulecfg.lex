%{
#include <string.h>
#include <stdlib.h>

#include "experimental/opt/parse/rulecfg.yy.h"

%}

%option nodefault
%option noyywrap

%%
[\t\r\a\v\b ]					; // spaces

\/\/.*							; // comments

symbol							{ return SYMBOL; }

\n								{ return NEWLINE; }

=>								{ return ARROW; }

\(								{ return LPAREN; }

\)								{ return RPAREN; }

,								{ return COMMA; }

=								{ return ASSIGN; }

\[ 								{ return LSB; }

\]								{ return RSB; }

\{ 								{ return LCB; }

\}								{ return RCB; }

groupdef						{ return GROUPDEF; }

group							{ return GROUP; }

shaper							{ return SHAPER; }

coorder							{ return COORDER; }

:								{ return COLON; }

\.\.\.							{ return VARIADIC; }

[a-zA-Z\_][a-zA-Z\_0-9]{0,31}	{
	strncpy(yylval.str_type, yytext, 32);
	return EL;
}

(0|[1-9][0-9]*)(\.\d+)?			{
	yylval.dec_type = atof(yytext);
	return NUMBER;
}

%%
