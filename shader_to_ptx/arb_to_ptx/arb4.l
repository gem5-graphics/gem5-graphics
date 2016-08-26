%{
#include <iostream>
using namespace std;
#include "y.tab.h"
%}
/*Regular expresions go here*/
alpha		[a-zA-Z]
numeric		[0-9]
hexalpha	[A-Fa-f]
alphanum	({alpha}|{numeric})
hexdigit	({numeric}|{hexalpha})
whitespace	[ \t]
newline		[\n]
anycharbutnewline [^\n]
anychar		({newline}|{anycharbutnewline})
pipe		\|

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?
fltliteral	[0-9]+\.[0-9]+

%%
[ \t]+	;
[\n]	;
"#"{anycharbutnewline}*	;
%{/* ignore version for now */%}
!!ARBvp1.0      {return VERTEX_PROGRAM;}
!!ARBfp1.0      {return FRAGMENT_PROGRAM;}
!!NVvp4.0       {return VERTEX_PROGRAM;}
!!NVfp4.0	{return FRAGMENT_PROGRAM;}

%{/* Instructions */%}
ABS	{return ABS;}
ADD	{return ADD;}
CMP	{return CMP;}
COS	{return COS;}
DIV	{return DIV;}
DP3	{return DP3;}
DP4	{return DP4;}
FLR     {return FLR;}
LG2	{return MAD;}
MAD	{return MAD;}
MOV	{return MOV;}
MOV_SAT {return MOV_SAT;}
MUL	{return MUL;}	
NOT	{return NOT;}	
OR	{return OR;}	
RCP	{return RCP;}
RSQ     {return RSQ;}
TEX	{return TEX;}
MAX     {return MAX;}
SLT     {return SLT;}

%{/* Others */%}
TEMP            {return TEMP;}
PARAM           {return PARAM;}
ATTRIB          {return ATTRIB;}
OUTPUT          {return OUTPUT;}
END             {return END;}
OPTION		{return OPTION;}
vertex          {return vertex;}
fragment        {return fragment;}
attrib          {return attrib;}
clip            {return clip;}
fogcoord        {return fogcoord;}
result          {return result;}
state           {return STATE;}
row             {return row;}
matrix          {return matrix;}
color           {return color;}
modelview	{return modelview;}
position	{return position;} 
transpose	{return transpose;}
projection	{return projection;}
normal          {return normal;}
inverse         {return inverse;}
light           {return light;}
ambient         {return ambient;}
diffuse         {return diffuse;}
program         {return program;}
local           {return local;}
half            {return half;}
material        {return material;}
front           {return front;}
lightmodel      {return lightmodel;}
mvp             {return mvp;}
texture         {return texture;}
texcoord        {return texcoord;}
SHORT   	{return SHORT;}
LONG    	{return LONG;}
INT     	{return INT;}
UINT     	{return UINT;}
FLOAT   	{return FLOAT;}
1D              {return oneD;}
2D              {return twoD;}
3D              {return threeD;}


F	{return 'F';}

%{/* xywx and rgba suffixes */%}
[rgba]+	{yylval.string_value = strdup(yytext); return rgbaSwizzleMask;}
[xyzw]+	{yylval.string_value = strdup(yytext); return xyzwSwizzleMask;}

";"	{return ';';}
"{"	{return '{';}
"}"	{return '}';}
"["	{return '[';}
"]"	{return ']';}
"="	{return '=';}
","	{return ',';}
"-"     {return '-';}
"+"     {return '+';}
":"	{return ':';}

[0-9]+\.[0-9]+ { yylval.float_value = atof(yytext); return floatConst; }
".."	{return TWO_DOTS;}
"."	{return '.';}
[0-9]+	{ yylval.integer_value = atoi(yytext); return intConst; }
{identifier}	{/*cout<<"found identifier ="<<yytext<<endl;*/ yylval.string_value = strdup(yytext); return STRING;}

%%
