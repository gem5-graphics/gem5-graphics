#include <GL/glew.h>

#include "register_states.h"

enum SASpecializations
{
#define CG_STATE_ASSIGNMENT_MACRO(state_name) \
    state_name,

#include "StateAssignmentIds.h"
};

#define ADD_TRUE_FALSE \
    cgAddStateEnumerant( state, "TRUE",  1 ); \
    cgAddStateEnumerant( state, "FALSE", 0 );

#define TEX_WRAP_STATE \
    cgAddStateEnumerant( state, "Repeat",              GL_REPEAT  ); \
    cgAddStateEnumerant( state, "Wrap",                GL_REPEAT ); \
    cgAddStateEnumerant( state, "Clamp",               GL_CLAMP ); \
    cgAddStateEnumerant( state, "ClampToEdge",         GL_CLAMP_TO_EDGE ); \
    cgAddStateEnumerant( state, "ClampToBorder",       GL_CLAMP_TO_BORDER ); \
    cgAddStateEnumerant( state, "Border",              GL_CLAMP_TO_BORDER ); \
    cgAddStateEnumerant( state, "MirroredRepeat",      GL_MIRRORED_REPEAT ); \
    cgAddStateEnumerant( state, "Mirror",              GL_MIRRORED_REPEAT ); \
    cgAddStateEnumerant( state, "MirrorClamp",         GL_MIRROR_CLAMP_EXT ); \
    cgAddStateEnumerant( state, "MirrorClampToEdge",   GL_MIRROR_CLAMP_TO_EDGE_EXT ); \
    cgAddStateEnumerant( state, "MirrorClampToBorder", GL_MIRROR_CLAMP_TO_BORDER_EXT ); \
    cgAddStateEnumerant( state, "MirrorOnce",          GL_MIRROR_CLAMP_EXT );

#define TEX_ARGS \
    cgAddStateEnumerant( state, "Constant",   0x00000006 ); /* D3DTA_CONSTANT */ \
    cgAddStateEnumerant( state, "Current",    0x00000001 ); /* D3DTA_CURRENT */ \
    cgAddStateEnumerant( state, "Diffuse",    0x00000000 ); /* D3DTA_DIFFUSE */ \
    cgAddStateEnumerant( state, "SelectMask", 0x0000000f ); /* D3DTA_SELECTMASK */ \
    cgAddStateEnumerant( state, "Specular",   0x00000004 ); /* D3DTA_SPECULAR */ \
    cgAddStateEnumerant( state, "Temp",       0x00000005 ); /* D3DTA_TEMP */ \
    cgAddStateEnumerant( state, "Texture",    0x00000002 ); /* D3DTA_TEXTURE */ \
    cgAddStateEnumerant( state, "TFactor",    0x00000003 ); /* D3DTA_TFACTOR */

#define TEX_OP \
    cgAddStateEnumerant( state, "Disable",                    1 ); /* D3DTOP_DISABLE */ \
    cgAddStateEnumerant( state, "SelectArg1",                 2 ); /* D3DTOP_SELECTARG1 */ \
    cgAddStateEnumerant( state, "SelectArg2",                 3 ); /* D3DTOP_SELECTARG2 */ \
    cgAddStateEnumerant( state, "Modulate",                   4 ); /* D3DTOP_MODULATE */ \
    cgAddStateEnumerant( state, "Modulate2x",                 5 ); /* D3DTOP_MODULATE2X */ \
    cgAddStateEnumerant( state, "Modulate4x",                 6 ); /* D3DTOP_MODULATE4X */ \
    cgAddStateEnumerant( state, "Add",                        7 ); /* D3DTOP_ADD */ \
    cgAddStateEnumerant( state, "AddSigned",                  8 ); /* D3DTOP_ADDSIGNED */ \
    cgAddStateEnumerant( state, "AddSigned2x",                9 ); /* D3DTOP_ADDSIGNED2X */ \
    cgAddStateEnumerant( state, "Subtract",                  10 ); /* D3DTOP_SUBTRACT */ \
    cgAddStateEnumerant( state, "AddSmooth",                 11 ); /* D3DTOP_ADDSMOOTH */ \
    cgAddStateEnumerant( state, "BlendDiffuseAlpha",         12 ); /* D3DTOP_BLENDDIFFUSEALPHA */ \
    cgAddStateEnumerant( state, "BlendTextureAlpha",         13 ); /* D3DTOP_BLENDTEXTUREALPHA */ \
    cgAddStateEnumerant( state, "BlendFactorAlpha",          14 ); /* D3DTOP_BLENDFACTORALPHA */ \
    cgAddStateEnumerant( state, "BlendTextureAlphaPM",       15 ); /* D3DTOP_BLENDTEXTUREALPHAPM */ \
    cgAddStateEnumerant( state, "BlendCurrentAlpha",         16 ); /* D3DTOP_BLENDCURRENTALPHA */ \
    cgAddStateEnumerant( state, "PreModulate",               17 ); /* D3DTOP_PREMODULATE */ \
    cgAddStateEnumerant( state, "ModulateAlpha_AddColor",    18 ); /* D3DTOP_MODULATEALPHA_ADDCOLOR */ \
    cgAddStateEnumerant( state, "ModulateColor_AddAlpha",    19 ); /* D3DTOP_MODULATECOLOR_ADDALPHA */ \
    cgAddStateEnumerant( state, "ModulateInvAlpha_AddColor", 20 ); /* D3DTOP_MODULATEINVALPHA_ADDCOLOR */ \
    cgAddStateEnumerant( state, "ModulateInvColor_AddAlpha", 21 ); /* D3DTOP_MODULATEINVCOLOR_ADDALPHA */ \
    cgAddStateEnumerant( state, "BumpEnvMap",                22 ); /* D3DTOP_BUMPENVMAP */ \
    cgAddStateEnumerant( state, "BumpEnvMapLuminance",       23 ); /* D3DTOP_BUMPENVMAPLUMINANCE */ \
    cgAddStateEnumerant( state, "DotProduct3",               24 ); /* D3DTOP_DOTPRODUCT3 */ \
    cgAddStateEnumerant( state, "MultiplyAdd",               25 ); /* D3DTOP_MULTIPLYADD */ \
    cgAddStateEnumerant( state, "Lerp",                      26 ); /* D3DTOP_LERP */

#define TEX_COORD_CAPS \
    cgAddStateEnumerant( state, "PassThru",                    0x00000000 ); /* D3DTSS_TCI_PASSTHRU */ \
    cgAddStateEnumerant( state, "CameraSpaceNormal",           0x00010000 ); /* D3DTSS_TCI_CAMERASPACENORMAL */ \
    cgAddStateEnumerant( state, "CameraSpacePosition",         0x00020000 ); /* D3DTSS_TCI_CAMERASPACEPOSITION */ \
    cgAddStateEnumerant( state, "CameraSpaceReflectionVector", 0x00030000 ); /* D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR */ \
    cgAddStateEnumerant( state, "SphereMap",                   0x00040000 ); /* D3DTSS_TCI_SPHEREMAP */

#define BLEND_MODE \
    cgAddStateEnumerant( state, "Zero",            1 ); /* D3DBLEND_ZERO */ \
    cgAddStateEnumerant( state, "One",             2 ); /* D3DBLEND_ONE  */ \
    cgAddStateEnumerant( state, "DestColor",       9 ); /* D3DBLEND_DESTCOLOR  */ \
    cgAddStateEnumerant( state, "InvDestColor",   10 ); /* D3DBLEND_INVDESTCOLOR */ \
    cgAddStateEnumerant( state, "SrcAlpha",        5 ); /* D3DBLEND_SRCALPHA */ \
    cgAddStateEnumerant( state, "InvSrcAlpha",     6 ); /* D3DBLEND_INVSRCALPHA */ \
    cgAddStateEnumerant( state, "DstAlpha",        7 ); /* D3DBLEND_DESTALPHA */ \
    cgAddStateEnumerant( state, "InvDestAlpha",    8 ); /* D3DBLEND_INVDESTALPHA */ \
    cgAddStateEnumerant( state, "SrcAlphaSat",    11 ); /* D3DBLEND_SRCALPHASAT */ \
    cgAddStateEnumerant( state, "SrcColor",        3 ); /* D3DBLEND_SRCCOLOR */ \
    cgAddStateEnumerant( state, "InvSrcColor",     4 ); /* D3DBLEND_INVSRCCOLOR */ \
    cgAddStateEnumerant( state, "BlendFactor",    14 ); /* D3DBLEND_BLENDFACTOR */ \
    cgAddStateEnumerant( state, "InvBlendFactor", 15 ); /* D3DBLEND_INVBLENDFACTOR */

void
RegisterStates(CGcontext context)
{
    CGstate state = RegisterState( "AlphaBlendEnable", CG_BOOL, 0, context, STATE_ALPHA_BLEND_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "AlphaFunc", CG_FLOAT2, 0, context, STATE_ALPHA_FUNC );
        cgAddStateEnumerant( state, "Never",    GL_NEVER    );
        cgAddStateEnumerant( state, "Less",     GL_LESS     );
        cgAddStateEnumerant( state, "LEqual",   GL_LEQUAL   );
        cgAddStateEnumerant( state, "Equal",    GL_EQUAL    );
        cgAddStateEnumerant( state, "Greater",  GL_GREATER  );
        cgAddStateEnumerant( state, "NotEqual", GL_NOTEQUAL );
        cgAddStateEnumerant( state, "GEqual",   GL_GEQUAL   );
        cgAddStateEnumerant( state, "Always",   GL_ALWAYS   );

    RegisterState( "AlphaRef", CG_FLOAT, 0, context, STATE_ALPHA_REF );

    state = RegisterState( "BlendOp", CG_INT, 0, context, STATE_BLEND_OP );
        cgAddStateEnumerant( state, "FuncAdd",                 GL_FUNC_ADD              );
        cgAddStateEnumerant( state, "FuncSubtract",            GL_FUNC_SUBTRACT         );
        cgAddStateEnumerant( state, "FuncReverseSubtract",     GL_FUNC_REVERSE_SUBTRACT );
        cgAddStateEnumerant( state, "Add",                     GL_FUNC_ADD              );
        cgAddStateEnumerant( state, "Subtract",                GL_FUNC_SUBTRACT         );
        cgAddStateEnumerant( state, "ReverseSubtract",         GL_FUNC_REVERSE_SUBTRACT );
        cgAddStateEnumerant( state, "Min",                     GL_MIN                   );
        cgAddStateEnumerant( state, "Max",                     GL_MAX                   );
        cgAddStateEnumerant( state, "LogicOp",                 GL_LOGIC_OP              );

    state = RegisterState( "BlendEquation", CG_INT, 0, context, STATE_BLEND_OP );
        cgAddStateEnumerant( state, "FuncAdd",                 GL_FUNC_ADD              );
        cgAddStateEnumerant( state, "FuncSubtract",            GL_FUNC_SUBTRACT         );
        cgAddStateEnumerant( state, "FuncReverseSubtract",     GL_FUNC_REVERSE_SUBTRACT );
        cgAddStateEnumerant( state, "Add",                     GL_FUNC_ADD              );
        cgAddStateEnumerant( state, "Subtract",                GL_FUNC_SUBTRACT         );
        cgAddStateEnumerant( state, "ReverseSubtract",         GL_FUNC_REVERSE_SUBTRACT );
        cgAddStateEnumerant( state, "Min",                     GL_MIN                   );
        cgAddStateEnumerant( state, "Max",                     GL_MAX                   );
        cgAddStateEnumerant( state, "LogicOp",                 GL_LOGIC_OP              );

    state = RegisterState( "BlendFunc", CG_INT2, 0, context, STATE_BLEND_FUNC ); // rgbaSrcFactor, rgbaDstFactor
        cgAddStateEnumerant( state, "Zero",                  GL_ZERO                         );
        cgAddStateEnumerant( state, "One",                   GL_ONE                          );
        cgAddStateEnumerant( state, "DestColor",             GL_DST_COLOR                    );
        cgAddStateEnumerant( state, "OneMinusDestColor",     GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "InvDestColor",          GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "SrcAlpha",              GL_SRC_ALPHA                    );
        cgAddStateEnumerant( state, "OneMinusSrcAlpha",      GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "InvSrcAlpha",           GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "DstAlpha",              GL_DST_ALPHA                    );
        cgAddStateEnumerant( state, "OneMinusDstAlpha",      GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "InvDestAlpha",          GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "SrcAlphaSaturate",      GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcAlphaSat",           GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcColor",              GL_SRC_COLOR                    );
        cgAddStateEnumerant( state, "OneMinusSrcColor",      GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "InvSrcColor",           GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "ConstantColor",         GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "BlendFactor",           GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "OneMinusConstantColor", GL_ONE_MINUS_CONSTANT_COLOR_EXT );
        cgAddStateEnumerant( state, "InvBlendFactor",        GL_ONE_MINUS_CONSTANT_COLOR_EXT );
        cgAddStateEnumerant( state, "ConstantAlpha",         GL_CONSTANT_ALPHA_EXT           );
        cgAddStateEnumerant( state, "OneMinusConstantAlpha", GL_ONE_MINUS_CONSTANT_ALPHA_EXT );

    state = RegisterState( "BlendFuncSeparate", CG_INT4, 0, context, STATE_BLEND_FUNC_SEPARATE );
        cgAddStateEnumerant( state, "Zero",                  GL_ZERO                         );
        cgAddStateEnumerant( state, "One",                   GL_ONE                          );
        cgAddStateEnumerant( state, "DestColor",             GL_DST_COLOR                    );
        cgAddStateEnumerant( state, "OneMinusDestColor",     GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "InvDestColor",          GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "SrcAlpha",              GL_SRC_ALPHA                    );
        cgAddStateEnumerant( state, "OneMinusSrcAlpha",      GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "InvSrcAlpha",           GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "DstAlpha",              GL_DST_ALPHA                    );
        cgAddStateEnumerant( state, "OneMinusDstAlpha",      GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "InvDestAlpha",          GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "SrcAlphaSaturate",      GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcAlphaSat",           GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcColor",              GL_SRC_COLOR                    );
        cgAddStateEnumerant( state, "OneMinusSrcColor",      GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "InvSrcColor",           GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "ConstantColor",         GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "BlendFactor",           GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "OneMinusConstantColor", GL_ONE_MINUS_CONSTANT_COLOR_EXT );
        cgAddStateEnumerant( state, "InvBlendFactor",        GL_ONE_MINUS_CONSTANT_COLOR_EXT );
        cgAddStateEnumerant( state, "ConstantAlpha",         GL_CONSTANT_ALPHA_EXT           );
        cgAddStateEnumerant( state, "OneMinusConstantAlpha", GL_ONE_MINUS_CONSTANT_ALPHA_EXT );

    state = RegisterState( "BlendEquationSeparate", CG_INT2, 0, context, STATE_BLEND_EQUATION_SEPARATE );
        cgAddStateEnumerant( state, "FuncAdd",             GL_FUNC_ADD      );
        cgAddStateEnumerant( state, "FuncSubtract",        GL_FUNC_SUBTRACT );
        cgAddStateEnumerant( state, "Min",                 GL_MIN           );
        cgAddStateEnumerant( state, "Max",                 GL_MAX           );
        cgAddStateEnumerant( state, "Add",                 GL_FUNC_ADD      );
        cgAddStateEnumerant( state, "Subtract",            GL_FUNC_SUBTRACT );
        cgAddStateEnumerant( state, "LogicOp",             GL_LOGIC_OP      );

    RegisterState( "BlendColor", CG_FLOAT4, 0, context, STATE_BLEND_COLOR );

    RegisterState( "ClearColor",   CG_FLOAT4, 0, context, STATE_CLEAR_COLOR );
    RegisterState( "ClearStencil", CG_INT, 0, context, STATE_CLEAR_STENCIL );
    RegisterState( "ClearDepth",   CG_FLOAT, 0, context, STATE_CLEAR_DEPTH );

    RegisterState( "ClipPlane", CG_FLOAT4, 6, context, STATE_CLIP_PLANE );
    state = RegisterState( "ClipPlaneEnable", CG_BOOL, 6, context, STATE_CLIP_PLANE_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "ColorWriteEnable", CG_BOOL4, 0, context, STATE_COLOR_WRITE_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "ColorMask", CG_BOOL4, 0, context, STATE_COLOR_WRITE_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "ColorVertex", CG_BOOL, 0, context, STATE_COLOR_VERTEX );
        ADD_TRUE_FALSE

    state = RegisterState( "ColorMaterial", CG_INT2, 0, context, STATE_COLOR_MATERIAL );
        cgAddStateEnumerant( state, "Emission",          GL_EMISSION            );
        cgAddStateEnumerant( state, "Emissive",          GL_EMISSION            );
        cgAddStateEnumerant( state, "Ambient",           GL_AMBIENT             );
        cgAddStateEnumerant( state, "Diffuse",           GL_DIFFUSE             );
        cgAddStateEnumerant( state, "Specular",          GL_SPECULAR            );
        cgAddStateEnumerant( state, "Front",             GL_FRONT               );
        cgAddStateEnumerant( state, "Back",              GL_BACK                );
        cgAddStateEnumerant( state, "FrontAndBack",      GL_FRONT_AND_BACK      );
        cgAddStateEnumerant( state, "AmbientAndDiffuse", GL_AMBIENT_AND_DIFFUSE );

    RegisterState( "ColorMatrix", CG_FLOAT4x4, 0, context, STATE_COLOR_MATRIX );
    RegisterState( "ColorTransform", CG_FLOAT4x4, 8, context, STATE_COLOR_MATRIX ); // not in GL

    state = RegisterState( "CullFace", CG_INT, 0, context, STATE_CULL_FACE );
        cgAddStateEnumerant( state, "Front",        GL_FRONT          );
        cgAddStateEnumerant( state, "Back",         GL_BACK           );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );

    state = RegisterState( "CullMode", CG_INT, 0, context, STATE_CULL_FACE );
        cgAddStateEnumerant( state, "None",    GL_NONE );
        cgAddStateEnumerant( state, "Front",   GL_FRONT );
        cgAddStateEnumerant( state, "Back",    GL_BACK );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );
        cgAddStateEnumerant( state, "CW",      GL_NONE );
        cgAddStateEnumerant( state, "CCW",     GL_NONE );

    RegisterState( "DepthBounds", CG_FLOAT2, 0, context, STATE_DEPTH_BOUNDS );
    RegisterState( "DepthBias", CG_FLOAT, 0, context, STATE_DEPTH_BIAS );

    state = RegisterState( "DepthFunc", CG_INT, 0, context, STATE_ZFUNC );
        cgAddStateEnumerant( state, "Never",        GL_NEVER    );
        cgAddStateEnumerant( state, "Less",         GL_LESS     );
        cgAddStateEnumerant( state, "LEqual",       GL_LEQUAL   );
        cgAddStateEnumerant( state, "LessEqual",    GL_LEQUAL   );
        cgAddStateEnumerant( state, "Equal",        GL_EQUAL    );
        cgAddStateEnumerant( state, "Greater",      GL_GREATER  );
        cgAddStateEnumerant( state, "NotEqual",     GL_NOTEQUAL );
        cgAddStateEnumerant( state, "GEqual",       GL_GEQUAL   );
        cgAddStateEnumerant( state, "GreaterEqual", GL_GEQUAL   );
        cgAddStateEnumerant( state, "Always",       GL_ALWAYS   );

    state = RegisterState( "ZFunc", CG_INT, 0, context, STATE_ZFUNC );
        cgAddStateEnumerant( state, "Never",        GL_NEVER    );
        cgAddStateEnumerant( state, "Less",         GL_LESS     );
        cgAddStateEnumerant( state, "LEqual",       GL_LEQUAL   );
        cgAddStateEnumerant( state, "LessEqual",    GL_LEQUAL   );
        cgAddStateEnumerant( state, "Equal",        GL_EQUAL    );
        cgAddStateEnumerant( state, "Greater",      GL_GREATER  );
        cgAddStateEnumerant( state, "NotEqual",     GL_NOTEQUAL );
        cgAddStateEnumerant( state, "GEqual",       GL_GEQUAL   );
        cgAddStateEnumerant( state, "GreaterEqual", GL_GEQUAL   );
        cgAddStateEnumerant( state, "Always",       GL_ALWAYS   );

    state = RegisterState( "DepthMask", CG_BOOL, 0, context, STATE_ZWRITE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "ZWriteEnable", CG_BOOL, 0, context, STATE_ZWRITE_ENABLE );
        ADD_TRUE_FALSE

    RegisterState( "DepthRange", CG_FLOAT2, 0, context, STATE_DEPTH_RANGE ); // only in GL

    RegisterState( "FogDistanceMode", CG_INT, 0, context, STATE_FOG_DISTANCE_MODE ); // only in GL
        cgAddStateEnumerant( state, "EyeRadial",        GL_EYE_RADIAL_NV         );
        cgAddStateEnumerant( state, "EyePlane",         GL_EYE_PLANE             );
        cgAddStateEnumerant( state, "EyePlaneAbsolute", GL_EYE_PLANE_ABSOLUTE_NV );

    state = RegisterState( "FogMode", CG_INT, 0, context, STATE_FOG_TABLE_MODE );
        cgAddStateEnumerant( state, "Linear", GL_LINEAR );
        cgAddStateEnumerant( state, "Exp",    GL_EXP    );
        cgAddStateEnumerant( state, "Exp2",   GL_EXP2   );

    state = RegisterState( "FogTableMode", CG_INT, 0, context, STATE_FOG_TABLE_MODE );
        cgAddStateEnumerant( state, "None",   GL_NONE );
        cgAddStateEnumerant( state, "Linear", GL_LINEAR );
        cgAddStateEnumerant( state, "Exp",    GL_EXP    );
        cgAddStateEnumerant( state, "Exp2",   GL_EXP2   );

    state = RegisterState( "IndexedVertexBlendEnable", CG_BOOL, 0, context, STATE_INDEXED_VERTEX_BLEND_ENABLE );
        ADD_TRUE_FALSE

    RegisterState( "FogDensity", CG_FLOAT, 0, context, STATE_FOG_DENSITY );
    RegisterState( "FogStart",   CG_FLOAT, 0, context, STATE_FOG_START );
    RegisterState( "FogEnd",     CG_FLOAT, 0, context, STATE_FOG_END );
    RegisterState( "FogColor",   CG_FLOAT4, 0, context, STATE_FOG_COLOR );

    RegisterState( "FragmentEnvParameter", CG_FLOAT4, 8192, context, STATE_FRAGMENT_ENV_PARAM );
    RegisterState( "FragmentLocalParameter", CG_FLOAT4, 8192, context, STATE_FRAGMENT_LOCAL_PARAM );

    state = RegisterState( "FogCoordSrc", CG_INT, 0, context, STATE_FOG_VERTEX_MODE );
        cgAddStateEnumerant( state, "FragmentDepth", GL_FRAGMENT_DEPTH_EXT );
        cgAddStateEnumerant( state, "FogCoord",      GL_FOG_COORDINATE_EXT );

    state = RegisterState( "FogVertexMode", CG_INT, 0, context, STATE_FOG_VERTEX_MODE );
        cgAddStateEnumerant( state, "None",   GL_NONE   );
        cgAddStateEnumerant( state, "Exp",    GL_EXP    );
        cgAddStateEnumerant( state, "Exp2",   GL_EXP2   );
        cgAddStateEnumerant( state, "Linear", GL_LINEAR );

    state = RegisterState( "FrontFace", CG_INT, 0, context, STATE_FRONT_FACE );
        cgAddStateEnumerant( state, "CW",  GL_CW  );
        cgAddStateEnumerant( state, "CCW", GL_CCW );

    // TODO: Add CullMode!

    RegisterState( "LightModelAmbient", CG_FLOAT4, 0, context, STATE_AMBIENT );
    RegisterState( "Ambient", CG_FLOAT4, 0, context, STATE_AMBIENT );

#define N_LIGHTS 8

    state = RegisterState( "LightingEnable", CG_BOOL, 0, context, STATE_LIGHTING_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "Lighting", CG_BOOL, 0, context, STATE_LIGHTING_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LightEnable",               CG_BOOL,   N_LIGHTS, context, STATE_LIGHT_ENABLE );
        ADD_TRUE_FALSE
    RegisterState( "LightAmbient",              CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_AMBIENT );
    RegisterState( "LightConstantAttenuation",  CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION0 );
    RegisterState( "LightAttenuation0",         CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION0 );
    RegisterState( "LightDiffuse",              CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_DIFFUSE );
    RegisterState( "LightLinearAttenuation",    CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION1 );
    RegisterState( "LightAttenuation1",         CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION1 );
    RegisterState( "LightPosition",             CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_POSITION );
    RegisterState( "LightQuadraticAttenuation", CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION2 );
    RegisterState( "LightAttenuation2",         CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_ATTENUATION2 );
    RegisterState( "LightSpecular",             CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_SPECULAR );
    RegisterState( "LightSpotCutoff",           CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_FALLOFF );
    RegisterState( "LightFalloff",              CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_FALLOFF );
    RegisterState( "LightSpotDirection",        CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_DIRECTION );
    RegisterState( "LightDirection",            CG_FLOAT4, N_LIGHTS, context, STATE_LIGHT_DIRECTION );
    RegisterState( "LightSpotExponent",         CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_SPOT_EXP );
    RegisterState( "LightPhi",                  CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_PHI );
    RegisterState( "LightRange",                CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_RANGE );
    RegisterState( "LightTheta",                CG_FLOAT,  N_LIGHTS, context, STATE_LIGHT_THETA );
    RegisterState( "LightType",                 CG_INT,    N_LIGHTS, context, STATE_LIGHT_TYPE );

    state = RegisterState( "LocalViewer", CG_BOOL, 0, context, STATE_LOCAL_VIEWER );
        ADD_TRUE_FALSE
    state = RegisterState( "MultiSampleAntialias", CG_BOOL, 0, context, STATE_MULTI_SAMPLE_AA );
        ADD_TRUE_FALSE
    RegisterState( "MultiSampleMask", CG_INT, 0, context, STATE_MULTI_SAMPLE_MASK );
    RegisterState( "PatchSegments", CG_FLOAT, 0, context, STATE_PATCH_SEGMENTS );
    RegisterState( "PointScale_A", CG_FLOAT, 0, context, STATE_POINT_SCALE_A );
    RegisterState( "PointScale_B", CG_FLOAT, 0, context, STATE_POINT_SCALE_B );
    RegisterState( "PointScale_C", CG_FLOAT, 0, context, STATE_POINT_SCALE_C );
    state = RegisterState( "PointScaleEnable", CG_BOOL, 0, context, STATE_POINT_SCALE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "RangeFogEnable", CG_BOOL, 0, context, STATE_RANGE_FOG_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "SpecularEnable", CG_BOOL, 0, context, STATE_SPECULAR_ENABLE );
        ADD_TRUE_FALSE
    RegisterState( "TweenFactor", CG_FLOAT, 0, context, STATE_TWEEN_FACTOR );
    RegisterState( "VertexBlend", CG_INT, 0, context, STATE_VERTEX_BLEND );

    RegisterState( "AmbientMaterialSource", CG_INT, 0, context, STATE_AMBIENT_MATERIAL_SOURCE );
    RegisterState( "DiffuseMaterialSource", CG_INT, 0, context, STATE_DIFFUSE_MATERIAL_SOURCE );
    RegisterState( "EmissiveMaterialSource", CG_INT, 0, context, STATE_EMISSIVE_MATERIAL_SOURCE );
    RegisterState( "SpecularMaterialSource", CG_INT, 0, context, STATE_SPECULAR_MATERIAL_SOURCE );

    state = RegisterState( "Clipping", CG_BOOL, 0, context, STATE_CLIPPING );
        ADD_TRUE_FALSE

    state = RegisterState( "LightModelColorControl", CG_INT, 0, context, STATE_LIGHT_MODEL_COLOR_CONTROL );
        cgAddStateEnumerant( state, "SingleColor",      GL_SINGLE_COLOR_EXT            );
        cgAddStateEnumerant( state, "SeparateSpecular", GL_SEPARATE_SPECULAR_COLOR_EXT );

    RegisterState( "LineStipple", CG_INT2, 0, context, STATE_LINE_STIPPLE );
    RegisterState( "LineWidth",   CG_FLOAT, 0, context, STATE_LINE_WIDTH );

    state = RegisterState( "LogicOp", CG_INT, 0, context, STATE_LOGIC_OP );
        cgAddStateEnumerant( state, "Clear",        GL_CLEAR         );
        cgAddStateEnumerant( state, "And",          GL_AND           );
        cgAddStateEnumerant( state, "AndReverse",   GL_AND_REVERSE   );
        cgAddStateEnumerant( state, "Copy",         GL_COPY          );
        cgAddStateEnumerant( state, "AndInverted",  GL_AND_INVERTED  );
        cgAddStateEnumerant( state, "Noop",         GL_NOOP          );
        cgAddStateEnumerant( state, "Xor",          GL_XOR           );
        cgAddStateEnumerant( state, "Or",           GL_OR            );
        cgAddStateEnumerant( state, "Nor",          GL_NOR           );
        cgAddStateEnumerant( state, "Equiv",        GL_EQUIV         );
        cgAddStateEnumerant( state, "Invert",       GL_INVERT        );
        cgAddStateEnumerant( state, "OrReverse",    GL_OR_REVERSE    );
        cgAddStateEnumerant( state, "CopyInverted", GL_COPY_INVERTED );
        cgAddStateEnumerant( state, "Nand",         GL_NAND          );
        cgAddStateEnumerant( state, "Set",          GL_SET           );


    RegisterState( "MaterialAmbient",   CG_FLOAT4, 0, context, STATE_MATERIAL_AMBIENT );
    RegisterState( "MaterialDiffuse",   CG_FLOAT4, 0, context, STATE_MATERIAL_DIFFUSE );
    RegisterState( "MaterialEmission",  CG_FLOAT4, 0, context, STATE_MATERIAL_EMISSIVE );
    RegisterState( "MaterialEmissive",  CG_FLOAT4, 0, context, STATE_MATERIAL_EMISSIVE );
    RegisterState( "MaterialShininess", CG_FLOAT,  0, context, STATE_MATERIAL_POWER );
    RegisterState( "MaterialPower",     CG_FLOAT,  0, context, STATE_MATERIAL_POWER );
    RegisterState( "MaterialSpecular",  CG_FLOAT4, 0, context, STATE_MATERIAL_SPECULAR );

    RegisterState( "ModelViewMatrix", CG_FLOAT4x4, 0, context, STATE_MODELVIEW_TRANSFORM );
    RegisterState( "ModelViewTransform", CG_FLOAT4x4, 0, context, STATE_MODELVIEW_TRANSFORM );

    RegisterState( "ViewTransform", CG_FLOAT4x4, 0, context, STATE_VIEW_TRANSFORM );
    RegisterState( "WorldTransform", CG_FLOAT4x4, 0, context, STATE_WORLD_TRANSFORM );

    RegisterState( "PointDistanceAttenuation", CG_FLOAT3, 0, context, STATE_POINT_DISTANCE_ATTENUATION );
    RegisterState( "PointFadeThresholdSize", CG_FLOAT, 0, context, STATE_POINT_FADE_THRESHOLD_SIZE );

    RegisterState( "PointSize", CG_FLOAT, 0, context, STATE_POINT_SIZE );
    RegisterState( "PointSizeMin", CG_FLOAT, 0, context, STATE_POINT_SIZE_MIN );
    RegisterState( "PointSizeMax", CG_FLOAT, 0, context, STATE_POINT_SIZE_MAX );

    RegisterState( "PointSpriteCoordOrigin", CG_INT, 0, context, STATE_POINT_SPRITE_COORD_ORIGIN );
        cgAddStateEnumerant( state, "LowerLeft", GL_LOWER_LEFT );
        cgAddStateEnumerant( state, "UpperLeft", GL_UPPER_LEFT );

    state = RegisterState( "PointSpriteCoordReplace", CG_BOOL, 4, context, STATE_POINT_SPRITE_COORD_REPLACE );
        ADD_TRUE_FALSE

    RegisterState( "PointSpriteRMode", CG_INT, 0, context, STATE_POINT_SPRITE_R_MODE );
        cgAddStateEnumerant( state, "Zero", GL_ZERO );
        cgAddStateEnumerant( state, "S",    GL_S    );
        cgAddStateEnumerant( state, "R",    GL_R    );

    state = RegisterState( "PolygonMode", CG_INT2, 0, context, STATE_FILL_MODE );
        cgAddStateEnumerant( state, "Front",        GL_FRONT          );
        cgAddStateEnumerant( state, "Back",         GL_BACK           );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );
        cgAddStateEnumerant( state, "Point",        GL_POINT          );
        cgAddStateEnumerant( state, "Line",         GL_LINE           );
        cgAddStateEnumerant( state, "Fill",         GL_FILL           );
        cgAddStateEnumerant( state, "Solid",        GL_FILL           );
        cgAddStateEnumerant( state, "Wireframe",    GL_LINE           );

    state = RegisterState( "FillMode", CG_INT2, 0, context, STATE_FILL_MODE );
        cgAddStateEnumerant( state, "Solid",       GL_FILL );
        cgAddStateEnumerant( state, "Wireframe",   GL_LINE );
        cgAddStateEnumerant( state, "Point",       GL_POINT );

    state = RegisterState( "LastPixel", CG_BOOL, 0, context, STATE_LAST_PIXEL );
        ADD_TRUE_FALSE

    RegisterState( "PolygonOffset", CG_FLOAT2, 0, context, STATE_POLYGON_OFFSET );

    RegisterState( "ProjectionMatrix", CG_FLOAT4x4, 0, context, STATE_PROJECTION_TRANSFORM );
    RegisterState( "ProjectionTransform", CG_FLOAT4x4, 0, context, STATE_PROJECTION_TRANSFORM );

    RegisterState( "Scissor", CG_INT4, 0, context, STATE_SCISSOR );

    state = RegisterState( "ShadeModel", CG_INT, 0, context, STATE_SHADE_MODE );
        cgAddStateEnumerant( state, "Flat",   GL_FLAT   );
        cgAddStateEnumerant( state, "Smooth", GL_SMOOTH );

    state = RegisterState( "ShadeMode", CG_INT, 0, context, STATE_SHADE_MODE );
        cgAddStateEnumerant( state, "Flat",    GL_FLAT   );
        cgAddStateEnumerant( state, "Smooth",  GL_SMOOTH );
        cgAddStateEnumerant( state, "Gouraud", GL_SMOOTH );
        cgAddStateEnumerant( state, "Phong",   GL_SMOOTH );

    RegisterState( "SlopScaleDepthBias", CG_FLOAT, 0, context, STATE_SLOPE_SCALE_DEPTH_BIAS );

    state = RegisterState( "DestBlend", CG_INT, 0, context, STATE_DEST_BLEND );
        cgAddStateEnumerant( state, "Zero",           GL_ZERO                         );
        cgAddStateEnumerant( state, "One",            GL_ONE                          );
        cgAddStateEnumerant( state, "DestColor",      GL_DST_COLOR                    );
        cgAddStateEnumerant( state, "InvDestColor",   GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "SrcAlpha",       GL_SRC_ALPHA                    );
        cgAddStateEnumerant( state, "InvSrcAlpha",    GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "DstAlpha",       GL_DST_ALPHA                    );
        cgAddStateEnumerant( state, "InvDestAlpha",   GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "SrcAlphaSat",    GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcColor",       GL_SRC_COLOR                    );
        cgAddStateEnumerant( state, "InvSrcColor",    GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "BlendFactor",    GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "InvBlendFactor", GL_ONE_MINUS_CONSTANT_COLOR_EXT );

    state = RegisterState( "SrcBlend", CG_INT, 0, context, STATE_SRC_BLEND );
        cgAddStateEnumerant( state, "Zero",           GL_ZERO                         );
        cgAddStateEnumerant( state, "One",            GL_ONE                          );
        cgAddStateEnumerant( state, "DestColor",      GL_DST_COLOR                    );
        cgAddStateEnumerant( state, "InvDestColor",   GL_ONE_MINUS_DST_COLOR          );
        cgAddStateEnumerant( state, "SrcAlpha",       GL_SRC_ALPHA                    );
        cgAddStateEnumerant( state, "InvSrcAlpha",    GL_ONE_MINUS_SRC_ALPHA          );
        cgAddStateEnumerant( state, "DstAlpha",       GL_DST_ALPHA                    );
        cgAddStateEnumerant( state, "InvDestAlpha",   GL_ONE_MINUS_DST_ALPHA          );
        cgAddStateEnumerant( state, "SrcAlphaSat",    GL_SRC_ALPHA_SATURATE           );
        cgAddStateEnumerant( state, "SrcColor",       GL_SRC_COLOR                    );
        cgAddStateEnumerant( state, "InvSrcColor",    GL_ONE_MINUS_SRC_COLOR          );
        cgAddStateEnumerant( state, "BlendFactor",    GL_CONSTANT_COLOR_EXT           );
        cgAddStateEnumerant( state, "InvBlendFactor", GL_ONE_MINUS_CONSTANT_COLOR_EXT );

    state = RegisterState( "StencilFunc", CG_INT3, 0, context, STATE_STENCIL_FUNC ); /* actually want 2*int, 1 uint */
        cgAddStateEnumerant( state, "Never",        GL_NEVER    );
        cgAddStateEnumerant( state, "Less",         GL_LESS     );
        cgAddStateEnumerant( state, "LEqual",       GL_LEQUAL   );
        cgAddStateEnumerant( state, "LessEqual",    GL_LEQUAL   );
        cgAddStateEnumerant( state, "Equal",        GL_EQUAL    );
        cgAddStateEnumerant( state, "Greater",      GL_GREATER  );
        cgAddStateEnumerant( state, "NotEqual",     GL_NOTEQUAL );
        cgAddStateEnumerant( state, "GEqual",       GL_GEQUAL   );
        cgAddStateEnumerant( state, "GreaterEqual", GL_GEQUAL   );
        cgAddStateEnumerant( state, "Always",       GL_ALWAYS   );

    RegisterState( "StencilMask", CG_INT, 0, context, STATE_STENCIL_WRITE_MASK ); /* need unsigned int */
    RegisterState( "StencilPass", CG_INT, 0, context, STATE_STENCIL_PASS );
    RegisterState( "StencilRef",  CG_INT, 0, context, STATE_STENCIL_REF );
    RegisterState( "StencilWriteMask", CG_INT, 0, context, STATE_STENCIL_WRITE_MASK );
    RegisterState( "StencilZFail", CG_INT, 0, context, STATE_STENCIL_ZFAIL );

    RegisterState( "TextureFactor", CG_INT, 0, context, STATE_TFACTOR );

    state = RegisterState( "StencilOp", CG_INT3, 0, context, STATE_STENCIL_OP );
        cgAddStateEnumerant( state, "Keep",     GL_KEEP          );
        cgAddStateEnumerant( state, "Zero",     GL_ZERO          );
        cgAddStateEnumerant( state, "Replace",  GL_REPLACE       );
        cgAddStateEnumerant( state, "Incr",     GL_INCR          );
        cgAddStateEnumerant( state, "Decr",     GL_DECR          );
        cgAddStateEnumerant( state, "Invert",   GL_INVERT        );
        cgAddStateEnumerant( state, "IncrWrap", GL_INCR_WRAP_EXT );
        cgAddStateEnumerant( state, "DecrWrap", GL_DECR_WRAP_EXT );
        cgAddStateEnumerant( state, "IncrSat",  GL_INCR          );
        cgAddStateEnumerant( state, "DecrSat",  GL_DECR          );

    // Like StencilFunc, StencilMask, & StencilOp but first arg is Front, Back, or FrontAndBack
    state = RegisterState( "StencilFuncSeparate", CG_INT4, 0, context, STATE_STENCIL_FUNC_SEPARATE );
        cgAddStateEnumerant( state, "Never",        GL_NEVER          );
        cgAddStateEnumerant( state, "Less",         GL_LESS           );
        cgAddStateEnumerant( state, "LEqual",       GL_LEQUAL         );
        cgAddStateEnumerant( state, "Equal",        GL_EQUAL          );
        cgAddStateEnumerant( state, "Greater",      GL_GREATER        );
        cgAddStateEnumerant( state, "NotEqual",     GL_NOTEQUAL       );
        cgAddStateEnumerant( state, "GEqual",       GL_GEQUAL         );
        cgAddStateEnumerant( state, "Always",       GL_ALWAYS         );
        cgAddStateEnumerant( state, "LessEqual",    GL_LEQUAL         );
        cgAddStateEnumerant( state, "GreaterEqual", GL_GEQUAL         );
        cgAddStateEnumerant( state, "Front",        GL_FRONT          );
        cgAddStateEnumerant( state, "Back",         GL_BACK           );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );


    state = RegisterState( "StencilMaskSeparate", CG_INT2, 0, context, STATE_STENCIL_MASK_SEPARATE );
        cgAddStateEnumerant( state, "Front",        GL_FRONT );
        cgAddStateEnumerant( state, "Back",         GL_BACK );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );

    state = RegisterState( "StencilOpSeparate", CG_INT4, 0, context, STATE_STENCIL_OP_SEPARATE );
        cgAddStateEnumerant( state, "Front",        GL_FRONT          );
        cgAddStateEnumerant( state, "Back",         GL_BACK           );
        cgAddStateEnumerant( state, "FrontAndBack", GL_FRONT_AND_BACK );
        cgAddStateEnumerant( state, "Keep",         GL_KEEP           );
        cgAddStateEnumerant( state, "Zero",         GL_ZERO           );
        cgAddStateEnumerant( state, "Replace",      GL_REPLACE        );
        cgAddStateEnumerant( state, "Incr",         GL_INCR           );
        cgAddStateEnumerant( state, "Decr",         GL_DECR           );
        cgAddStateEnumerant( state, "Invert",       GL_INVERT         );
        cgAddStateEnumerant( state, "IncrWrap",     GL_INCR_WRAP_EXT  );
        cgAddStateEnumerant( state, "DecrWrap",     GL_DECR_WRAP_EXT  );
        cgAddStateEnumerant( state, "IncrSat",      GL_INCR           );
        cgAddStateEnumerant( state, "DecrSat",      GL_DECR           );

    state = RegisterState( "TexGenSMode", CG_INT, 4, context, STATE_TEXGEN_S_MODE );
        cgAddStateEnumerant( state, "ObjectLinear",  GL_OBJECT_LINEAR      );
        cgAddStateEnumerant( state, "EyeLinear",     GL_EYE_LINEAR         );
        cgAddStateEnumerant( state, "SphereMap",     GL_SPHERE_MAP         );
        cgAddStateEnumerant( state, "ReflectionMap", GL_REFLECTION_MAP_ARB );
        cgAddStateEnumerant( state, "NormalMap",     GL_NORMAL_MAP_ARB     );

    RegisterState( "TexGenSObjectPlane", CG_FLOAT4, 4, context, STATE_TEXGEN_S_OBJECT_PLANE );
    RegisterState( "TexGenSEyePlane",    CG_FLOAT4, 4, context, STATE_TEXGEN_S_EYE_PLANE );

    state = RegisterState( "TexGenTMode", CG_INT, 4, context, STATE_TEXGEN_T_MODE );
        cgAddStateEnumerant( state, "ObjectLinear",  GL_OBJECT_LINEAR      );
        cgAddStateEnumerant( state, "EyeLinear",     GL_EYE_LINEAR         );
        cgAddStateEnumerant( state, "SphereMap",     GL_SPHERE_MAP         );
        cgAddStateEnumerant( state, "ReflectionMap", GL_REFLECTION_MAP_ARB );
        cgAddStateEnumerant( state, "NormalMap",     GL_NORMAL_MAP_ARB     );

    RegisterState( "TexGenTObjectPlane", CG_FLOAT4, 4, context, STATE_TEXGEN_T_OBJECT_PLANE );
    RegisterState( "TexGenTEyePlane",    CG_FLOAT4, 4, context, STATE_TEXGEN_T_EYE_PLANE );

    state = RegisterState( "TexGenRMode", CG_INT, 4, context, STATE_TEXGEN_R_MODE );
        cgAddStateEnumerant( state, "ObjectLinear",  GL_OBJECT_LINEAR      );
        cgAddStateEnumerant( state, "EyeLinear",     GL_EYE_LINEAR         );
        cgAddStateEnumerant( state, "SphereMap",     GL_SPHERE_MAP         );
        cgAddStateEnumerant( state, "ReflectionMap", GL_REFLECTION_MAP_ARB );
        cgAddStateEnumerant( state, "NormalMap",     GL_NORMAL_MAP_ARB     );

    RegisterState( "TexGenRObjectPlane", CG_FLOAT4, 4, context, STATE_TEXGEN_R_OBJECT_PLANE );
    RegisterState( "TexGenREyePlane",    CG_FLOAT4, 4, context, STATE_TEXGEN_R_EYE_PLANE );

    state = RegisterState( "TexGenQMode", CG_INT, 4, context, STATE_TEXGEN_Q_MODE );
        cgAddStateEnumerant( state, "ObjectLinear",  GL_OBJECT_LINEAR      );
        cgAddStateEnumerant( state, "EyeLinear",     GL_EYE_LINEAR         );
        cgAddStateEnumerant( state, "SphereMap",     GL_SPHERE_MAP         );
        cgAddStateEnumerant( state, "ReflectionMap", GL_REFLECTION_MAP_ARB );
        cgAddStateEnumerant( state, "NormalMap",     GL_NORMAL_MAP_ARB     );

    RegisterState( "TexGenQObjectPlane", CG_FLOAT4, 4, context, STATE_TEXGEN_Q_OBJECT_PLANE );
    RegisterState( "TexGenQEyePlane",    CG_FLOAT4, 4, context, STATE_TEXGEN_Q_EYE_PLANE );

    RegisterState( "TextureEnvColor", CG_FLOAT4, 4, context, STATE_TEXTURE_ENV_COLOR );

    state = RegisterState( "TextureEnvMode", CG_INT, 4, context, STATE_TEXTURE_ENV_MODE );
        cgAddStateEnumerant( state, "Modulate", GL_MODULATE );
        cgAddStateEnumerant( state, "Decal",    GL_DECAL    );
        cgAddStateEnumerant( state, "Blend",    GL_BLEND    );
        cgAddStateEnumerant( state, "Replace",  GL_REPLACE  );
        cgAddStateEnumerant( state, "Add",      GL_ADD      );

    RegisterState( "Texture1D",        CG_SAMPLER1D, 16, context, STATE_TEXTURE_1D );
    RegisterState( "Texture2D",        CG_SAMPLER2D, 16, context, STATE_TEXTURE_2D );
    RegisterState( "Texture3D",        CG_SAMPLER3D, 16, context, STATE_TEXTURE_3D );
    RegisterState( "TextureRectangle", CG_SAMPLERRECT, 16, context, STATE_TEXTURE_RECT );
    RegisterState( "TextureCubeMap",   CG_SAMPLERCUBE, 16, context, STATE_TEXTURE_CUBE_MAP );

    state = RegisterState( "Texture1DEnable",        CG_BOOL, 16, context, STATE_TEXTURE_1D_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "Texture2DEnable",        CG_BOOL, 16, context, STATE_TEXTURE_2D_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "Texture3DEnable",        CG_BOOL, 16, context, STATE_TEXTURE_3D_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "TextureRectangleEnable", CG_BOOL, 16, context, STATE_TEXTURE_RECT_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "TextureCubeMapEnable",   CG_BOOL, 16, context, STATE_TEXTURE_CUBE_MAP_ENABLE );
        ADD_TRUE_FALSE

    //REGISTER_ARRAY_STATE(TextureTransform, CG_FLOAT4x4, 8);  // arrays of matrix state currently don't work
    RegisterState( "TextureTransform", CG_FLOAT4x4, 0, context, STATE_TEXTURE_TRANSFORM );
    RegisterState( "TextureMatrix", CG_FLOAT4x4, 0, context, STATE_TEXTURE_TRANSFORM );

    // See comment above FragmentProgramParameter for discussion of why
    // 8192 for the array size here...
    RegisterState( "VertexEnvParameter",   CG_FLOAT4, 8192, context, STATE_VERTEX_ENV_PARAM );
    RegisterState( "VertexLocalParameter", CG_FLOAT4, 8192, context, STATE_VERTEX_LOCAL_PARAM );

    // stuff that's just glEnable/glDisable
    state = RegisterState( "AlphaTestEnable",             CG_BOOL, 0, context, STATE_ALPHA_TEST_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "AutoNormalEnable",            CG_BOOL, 0, context, STATE_AUTO_NORMAL_ENABLE );       // for evaluators
        ADD_TRUE_FALSE
    state = RegisterState( "BlendEnable",                 CG_BOOL, 0, context, STATE_BLEND_ENABLE );             // vs AlphaBlendEnable
        ADD_TRUE_FALSE
    state = RegisterState( "ColorLogicOpEnable",          CG_BOOL, 0, context, STATE_COLOR_LOGIC_OP_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "CullFaceEnable",              CG_BOOL, 0, context, STATE_CULL_FACE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "DepthBoundsEnable",           CG_BOOL, 0, context, STATE_DEPTH_BOUNDS_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "DepthClampEnable",            CG_BOOL, 0, context, STATE_DEPTH_CLAMP_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "DepthTestEnable",             CG_BOOL, 0, context, STATE_ZENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "ZEnable",                     CG_BOOL, 0, context, STATE_ZENABLE ); // D3D
        ADD_TRUE_FALSE
    state = RegisterState( "DitherEnable",                CG_BOOL, 0, context, STATE_DITHER_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "FogEnable",                   CG_BOOL, 0, context, STATE_FOG_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LightModelLocalViewerEnable", CG_BOOL, 0, context, STATE_LIGHT_MODEL_LOCAL_VIEWER_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LightModelTwoSideEnable",     CG_BOOL, 0, context, STATE_LIGHT_MODEL_TWO_SIDE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LineSmoothEnable",            CG_BOOL, 0, context, STATE_LINE_SMOOTH_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LineStippleEnable",           CG_BOOL, 0, context, STATE_LINE_STIPPLE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "LogicOpEnable",               CG_BOOL, 0, context, STATE_LOGIC_OP_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "MultisampleEnable",           CG_BOOL, 0, context, STATE_MULTI_SAMPLE_AA );
        ADD_TRUE_FALSE
    state = RegisterState( "NormalizeEnable",             CG_BOOL, 0, context, STATE_NORMALIZE_NORMALS );
        ADD_TRUE_FALSE
    state = RegisterState( "NormalizeNormals",            CG_BOOL, 0, context, STATE_NORMALIZE_NORMALS );
        ADD_TRUE_FALSE
    state = RegisterState( "PointSmoothEnable",           CG_BOOL, 0, context, STATE_POINT_SMOOTH_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PointSpriteEnable",           CG_BOOL, 0, context, STATE_POINT_SPRITE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PolygonOffsetFillEnable",     CG_BOOL, 0, context, STATE_POLYGON_OFFSET_FILL_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PolygonOffsetLineEnable",     CG_BOOL, 0, context, STATE_POLYGON_OFFSET_LINE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PolygonOffsetPointEnable",    CG_BOOL, 0, context, STATE_POLYGON_OFFSET_POINT_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PolygonSmoothEnable",         CG_BOOL, 0, context, STATE_POLYGON_SMOOTH_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "PolygonStippleEnable",        CG_BOOL, 0, context, STATE_POLYGON_STIPPLE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "RescaleNormalEnable",         CG_BOOL, 0, context, STATE_RESCALE_NORMAL_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "SampleAlphaToCoverageEnable", CG_BOOL, 0, context, STATE_SAMPLE_ALPHA_TO_COV_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "SampleAlphaToOneEnable",      CG_BOOL, 0, context, STATE_SAMPLE_ALPHA_TO_ONE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "SampleCoverageEnable",        CG_BOOL, 0, context, STATE_SAMPLE_COVERAGE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "ScissorTestEnable",           CG_BOOL, 0, context, STATE_SCISSOR_TEST_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "StencilTestEnable",           CG_BOOL, 0, context, STATE_STENCIL_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "StencilEnable",               CG_BOOL, 0, context, STATE_STENCIL_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "StencilTestTwoSideEnable",    CG_BOOL, 0, context, STATE_STENCIL_TEST_TWO_SIDE_ENABLE );
        ADD_TRUE_FALSE

    RegisterState( "StencilFail", CG_INT, 0, context, STATE_STENCIL_FAIL );

    state = RegisterState( "TexGenSEnable", CG_BOOL, 4, context, STATE_TEXGEN_S_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "TexGenTEnable", CG_BOOL, 4, context, STATE_TEXGEN_T_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "TexGenREnable", CG_BOOL, 4, context, STATE_TEXGEN_R_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "TexGenQEnable", CG_BOOL, 4, context, STATE_TEXGEN_Q_ENABLE );
        ADD_TRUE_FALSE

    RegisterState( "Wrap0", CG_INT, 0, context, STATE_WRAP0 );
    RegisterState( "Wrap1", CG_INT, 0, context, STATE_WRAP1 );
    RegisterState( "Wrap2", CG_INT, 0, context, STATE_WRAP2 );
    RegisterState( "Wrap3", CG_INT, 0, context, STATE_WRAP3 );
    RegisterState( "Wrap4", CG_INT, 0, context, STATE_WRAP4 );
    RegisterState( "Wrap5", CG_INT, 0, context, STATE_WRAP5 );
    RegisterState( "Wrap6", CG_INT, 0, context, STATE_WRAP6 );
    RegisterState( "Wrap7", CG_INT, 0, context, STATE_WRAP7 );
    RegisterState( "Wrap8", CG_INT, 0, context, STATE_WRAP8 );
    RegisterState( "Wrap9", CG_INT, 0, context, STATE_WRAP9 );
    RegisterState( "Wrap10", CG_INT, 0, context, STATE_WRAP10 );
    RegisterState( "Wrap11", CG_INT, 0, context, STATE_WRAP11 );
    RegisterState( "Wrap12", CG_INT, 0, context, STATE_WRAP12 );
    RegisterState( "Wrap13", CG_INT, 0, context, STATE_WRAP13 );
    RegisterState( "Wrap14", CG_INT, 0, context, STATE_WRAP14 );
    RegisterState( "Wrap15", CG_INT, 0, context, STATE_WRAP15 );

    state = RegisterState( "VertexProgramPointSizeEnable", CG_BOOL, 0, context, STATE_VERTEX_PROGRAM_PSIZE_ENABLE );
        ADD_TRUE_FALSE
    state = RegisterState( "VertexProgramTwoSideEnable", CG_BOOL, 0, context, STATE_VERTEX_PROGRAM_TWO_SIDE_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "TessellationControlProgram", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "TessellationEvaluationProgram", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "GeometryProgram", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "VertexProgram", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "FragmentProgram", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "TessellationControlShader", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "TessellationEvaluationShader", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "GeometryShader", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "VertexShader", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    state = RegisterState( "PixelShader", CG_PROGRAM_TYPE, 0, context, STATE_SHADER );
        cgAddStateEnumerant( state, "Null", 0 );

    // D3D9 FX states

    // these are really M x N array of floats according to the docs
    RegisterState( "PixelShaderConstant",  CG_FLOAT4x4, 224, context, STATE_PIXEL_SHADER_CONSTANT );
    RegisterState( "VertexShaderConstant", CG_FLOAT4x4, 256, context, STATE_VERTEX_SHADER_CONSTANT );

    RegisterState( "PixelShaderConstant1", CG_FLOAT4,   224, context, STATE_PIXEL_SHADER_CONSTANT1 );
    RegisterState( "PixelShaderConstant2", CG_FLOAT4x2, 224, context, STATE_PIXEL_SHADER_CONSTANT2 );
    RegisterState( "PixelShaderConstant3", CG_FLOAT4x3, 224, context, STATE_PIXEL_SHADER_CONSTANT3 );
    RegisterState( "PixelShaderConstant4", CG_FLOAT4x4, 224, context, STATE_PIXEL_SHADER_CONSTANT4 );

    RegisterState( "VertexShaderConstant1", CG_FLOAT4,   256, context, STATE_VERTEX_SHADER_CONSTANT1 );
    RegisterState( "VertexShaderConstant2", CG_FLOAT4x2, 256, context, STATE_VERTEX_SHADER_CONSTANT2 );
    RegisterState( "VertexShaderConstant3", CG_FLOAT4x3, 256, context, STATE_VERTEX_SHADER_CONSTANT3 );
    RegisterState( "vertexShaderConstant4", CG_FLOAT4x4, 256, context, STATE_VERTEX_SHADER_CONSTANT4 );

    state = RegisterState( "PixelShaderConstantB", CG_BOOL4,  224, context, STATE_PIXEL_SHADER_CONSTANT_B );
        ADD_TRUE_FALSE
    RegisterState( "PixelShaderConstantI", CG_INT4,   224, context, STATE_PIXEL_SHADER_CONSTANT_I );
    RegisterState( "PixelShaderConstantF", CG_FLOAT4, 224, context, STATE_PIXEL_SHADER_CONSTANT_F );

    state = RegisterState( "VertexShaderConstantB", CG_BOOL4,  256, context, STATE_VERTEX_SHADER_CONSTANT_B );
        ADD_TRUE_FALSE
    RegisterState( "VertexShaderConstantI", CG_INT4,   256, context, STATE_VERTEX_SHADER_CONSTANT_I );
    RegisterState( "VertexShaderConstantF", CG_FLOAT4, 256, context, STATE_VERTEX_SHADER_CONSTANT_F );

    RegisterState( "Texture", CG_TEXTURE, 16, context, STATE_TEXTURE_PASS );
    RegisterState( "Sampler", CG_SAMPLER, 16, context, STATE_SAMPLER );

    state = RegisterState( "AddressU",      CG_INT, 16, context, STATE_ADDRESS_U_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "AddressV",      CG_INT, 16, context, STATE_ADDRESS_V_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "AddressW",      CG_INT, 16, context, STATE_ADDRESS_W_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "BorderColor",   CG_INT, 16, context, STATE_BORDER_COLOR_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "MaxAnisotropy", CG_INT, 16, context, STATE_MAX_ANISOTROPY_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "MaxMipLevel",   CG_INT, 16, context, STATE_MAX_MIP_LEVEL_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "MinFilter",     CG_INT, 16, context, STATE_MIN_FILTER_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "MagFilter",     CG_INT, 16, context, STATE_MAG_FILTER_16 );
        TEX_WRAP_STATE;
    state = RegisterState( "MipFilter",     CG_INT, 16, context, STATE_MIP_FILTER_16 );
        TEX_WRAP_STATE;
        cgAddStateEnumerant( state, "None",          0 ); /* D3DTEXF_NONE */
        cgAddStateEnumerant( state, "Point",         1 ); /* D3DTEXF_POINT */
        cgAddStateEnumerant( state, "Linear",        2 ); /* D3DTEXF_LINEAR */
        cgAddStateEnumerant( state, "Anisotropic",   3 ); /* D3DTEXF_ANISOTROPIC */
        cgAddStateEnumerant( state, "PyramidalQuad", 6 ); /* D3DTEXF_PYRAMIDALQUAD */
        cgAddStateEnumerant( state, "GaussianQuad",  7 ); /* D3DTEXF_GAUSSIANQUAD */
    state = RegisterState( "MipMapLodBias", CG_INT, 16, context, STATE_MIPMAP_LOD_BIAS_16 );
        TEX_WRAP_STATE;

    state = RegisterState( "ColorWriteEnable1",   CG_BOOL4, 0, context, STATE_COLOR_WRITE_ENABLE_1 );
        ADD_TRUE_FALSE
    state = RegisterState( "ColorWriteEnable2",   CG_BOOL4, 0, context, STATE_COLOR_WRITE_ENABLE_2 );
        ADD_TRUE_FALSE
    state = RegisterState( "ColorWriteEnable3",   CG_BOOL4, 0, context, STATE_COLOR_WRITE_ENABLE_3 );
        ADD_TRUE_FALSE
    state = RegisterState( "TwoSidedStencilMode", CG_BOOL,  0, context, STATE_STENCIL_TEST_TWO_SIDE_ENABLE );
        ADD_TRUE_FALSE

    state = RegisterState( "BlendOpAlpha", CG_INT, 0, context, STATE_BLEND_OP_ALPHA );
        cgAddStateEnumerant( state, "Add",         1 ); /* D3DBLENDOP_ADD */
        cgAddStateEnumerant( state, "Subtract",    2 ); /* D3DBLENDOP_SUBTRACT */
        cgAddStateEnumerant( state, "RevSubtract", 3 ); /* D3DBLENDOP_REVSUBTRACT */
        cgAddStateEnumerant( state, "Min",         4 ); /* D3DBLENDOP_MIN */
        cgAddStateEnumerant( state, "Max",         5 ); /* D3DBLENDOP_MAX */

    state = RegisterState( "SrcBlendAlpha", CG_INT, 0, context, STATE_SRC_BLEND_ALPHA );
        BLEND_MODE;

    state = RegisterState( "DestBlendAlpha", CG_INT, 0, context, STATE_DEST_BLEND_ALPHA );
        BLEND_MODE;

    state = RegisterState( "SeparateAlphaBlendEnable", CG_BOOL, 0, context, STATE_SEPARATE_ALPHA_BLEND_ENABLE );
        ADD_TRUE_FALSE

}

void
RegisterSamplerStates(CGcontext context)
{
    CGstate state;

    RegisterState( "Texture", CG_TEXTURE, -1, context, STATE_TEXTURE );

    // -1 means create sampler state
    state = RegisterState( "AddressU", CG_INT, -1, context, STATE_ADDRESS_U );
        TEX_WRAP_STATE;

    state = RegisterState( "AddressV", CG_INT, -1, context, STATE_ADDRESS_V );
        TEX_WRAP_STATE;

    state = RegisterState( "AddressW", CG_INT, -1, context, STATE_ADDRESS_W );
        TEX_WRAP_STATE;

    state = RegisterState( "WrapS", CG_INT, -1, context, STATE_ADDRESS_U );
        TEX_WRAP_STATE;

    state = RegisterState( "WrapT", CG_INT, -1, context, STATE_ADDRESS_V );
        TEX_WRAP_STATE;

    state = RegisterState( "WrapR", CG_INT, -1, context, STATE_ADDRESS_W );
        TEX_WRAP_STATE;

    state = RegisterState( "MipFilter", CG_INT, -1, context, STATE_MIP_FILTER );
        cgAddStateEnumerant( state, "None",                 0); // D3DTEXF_NONE
        cgAddStateEnumerant( state, "Point",                1); // D3DTEXF_POINT
        cgAddStateEnumerant( state, "Linear",               2); // D3DTEXF_LINEAR
        cgAddStateEnumerant( state, "Anisotropic",          3); // D3DTEXF_ANISOTROPIC
        cgAddStateEnumerant( state, "PyramidalQuad",        6); // D3DTEXF_PYRAMIDALQUAD
        cgAddStateEnumerant( state, "GaussianQuad",         7); // D3DTEXF_GAUSSIANQUAD

    RegisterState( "MipMapLodBias", CG_FLOAT, -1, context, STATE_MIPMAP_LOD_BIAS );
    RegisterState( "LODBias", CG_FLOAT, -1, context, STATE_MIPMAP_LOD_BIAS );

    RegisterState( "SRGBTexture", CG_FLOAT, -1, context, STATE_SRGB_TEXTURE );

    state = RegisterState( "MinFilter", CG_INT, -1, context, STATE_MIN_FILTER );
        cgAddStateEnumerant( state, "Point",                GL_NEAREST                );
        cgAddStateEnumerant( state, "Nearest",              GL_NEAREST                );
        cgAddStateEnumerant( state, "Linear",               GL_LINEAR                 );
        cgAddStateEnumerant( state, "LinearMipMapNearest",  GL_LINEAR_MIPMAP_NEAREST  );
        cgAddStateEnumerant( state, "NearestMipMapNearest", GL_NEAREST_MIPMAP_NEAREST );
        cgAddStateEnumerant( state, "NearestMipMapLinear",  GL_NEAREST_MIPMAP_LINEAR  );
        cgAddStateEnumerant( state, "LinearMipMapLinear",   GL_LINEAR_MIPMAP_LINEAR   );
        cgAddStateEnumerant( state, "None",                 0); // D3DTEXF_NONE
        cgAddStateEnumerant( state, "Point",                1); // D3DTEXF_POINT
        cgAddStateEnumerant( state, "Linear",               2); // D3DTEXF_LINEAR
        cgAddStateEnumerant( state, "Anisotropic",          3); // D3DTEXF_ANISOTROPIC
        cgAddStateEnumerant( state, "PyramidalQuad",        6); // D3DTEXF_PYRAMIDALQUAD
        cgAddStateEnumerant( state, "GaussianQuad",         7); // D3DTEXF_GAUSSIANQUAD


    state = RegisterState( "MagFilter", CG_INT, -1, context, STATE_MAG_FILTER );
        cgAddStateEnumerant( state, "Point",   GL_NEAREST );
        cgAddStateEnumerant( state, "Nearest", GL_NEAREST );
        cgAddStateEnumerant( state, "Linear",  GL_LINEAR  );


    RegisterState( "BorderColor", CG_FLOAT4, -1, context, STATE_BORDER_COLOR );

    RegisterState( "MinMipLevel",   CG_FLOAT, -1, context, STATE_MIN_MIP_LEVEL );
    RegisterState( "MaxMipLevel",   CG_FLOAT, -1, context, STATE_MAX_MIP_LEVEL );
    RegisterState( "MaxAnisotropy", CG_FLOAT, -1, context, STATE_MAX_ANISOTROPY );

    state = RegisterState( "DepthMode", CG_INT, -1, context, STATE_DEPTH_MODE );
        cgAddStateEnumerant( state, "Alpha",     GL_ALPHA     );
        cgAddStateEnumerant( state, "Intensity", GL_INTENSITY );
        cgAddStateEnumerant( state, "Luminance", GL_LUMINANCE );

    state = RegisterState( "CompareMode", CG_INT, -1, context, STATE_COMPARE_MODE );
        cgAddStateEnumerant( state, "None",              GL_NONE                     );
        cgAddStateEnumerant( state, "CompareRToTexture", GL_COMPARE_R_TO_TEXTURE_ARB );

    state = RegisterState( "CompareFunc", CG_INT, -1, context, STATE_COMPARE_FUNC );
        cgAddStateEnumerant( state, "Never",    GL_NEVER    );
        cgAddStateEnumerant( state, "Less",     GL_LESS     );
        cgAddStateEnumerant( state, "LEqual",   GL_LEQUAL   );
        cgAddStateEnumerant( state, "Equal",    GL_EQUAL    );
        cgAddStateEnumerant( state, "Greater",  GL_GREATER  );
        cgAddStateEnumerant( state, "NotEqual", GL_NOTEQUAL );
        cgAddStateEnumerant( state, "GEqual",   GL_GEQUAL   );
        cgAddStateEnumerant( state, "Always",   GL_ALWAYS   );

    state = RegisterState( "GenerateMipmap", CG_BOOL, -1, context, STATE_GENERATE_MIPMAP );
        ADD_TRUE_FALSE

    state = RegisterState( "AlphaOp", CG_INT, 8, context, STATE_ALPHA_OP );
        TEX_OP;
    state = RegisterState( "ColorOp",   CG_INT, 8, context, STATE_COLOR_OP );
        TEX_OP;

    state = RegisterState( "AlphaArg0", CG_INT, 8, context, STATE_ALPHA_ARG0 );
        TEX_ARGS;
    state = RegisterState( "AlphaArg1", CG_INT, 8, context, STATE_ALPHA_ARG1 );
        TEX_ARGS;
    state = RegisterState( "AlphaArg2", CG_INT, 8, context, STATE_ALPHA_ARG2 );
        TEX_ARGS;
    state = RegisterState( "ColorArg0", CG_INT, 8, context, STATE_COLOR_ARG0 );
        TEX_ARGS;
    state = RegisterState( "ColorArg1", CG_INT, 8, context, STATE_COLOR_ARG1 );
        TEX_ARGS;
    state = RegisterState( "ColorArg2", CG_INT, 8, context, STATE_COLOR_ARG2 );
        TEX_ARGS;

    state = RegisterState( "BumpEnvLScale", CG_FLOAT, 8, context, STATE_BUMP_ENV_LSCALE );
        TEX_COORD_CAPS;
    state = RegisterState( "BumpEnvLOffset", CG_FLOAT, 8, context, STATE_BUMP_ENV_LOFFSET );
        TEX_COORD_CAPS;

    RegisterState( "BumpEnvMat00", CG_FLOAT, 8, context, STATE_BUMP_ENV_MAT00 );
    RegisterState( "BumpEnvMat01", CG_FLOAT, 8, context, STATE_BUMP_ENV_MAT01 );
    RegisterState( "BumpEnvMat10", CG_FLOAT, 8, context, STATE_BUMP_ENV_MAT10 );
    RegisterState( "BumpEnvMat11", CG_FLOAT, 8, context, STATE_BUMP_ENV_MAT11 );

    state = RegisterState( "ResultArg", CG_INT, 8, context, STATE_RESULT_ARG );
        TEX_ARGS;

    state = RegisterState( "TexCoordIndex", CG_INT, 8, context, STATE_TEX_COORD_INDEX );
        TEX_COORD_CAPS;

    state = RegisterState( "TextureTransformFlags", CG_INT, 8, context, STATE_TEXTURE_TRANSFORM_FLAGS );
        cgAddStateEnumerant( state, "Disable",     0); // D3DTTFF_DISABLE );
        cgAddStateEnumerant( state, "Count1",      1); // D3DTTFF_COUNT1 );
        cgAddStateEnumerant( state, "Count2",      2); // D3DTTFF_COUNT2 );
        cgAddStateEnumerant( state, "Count3",      3); // D3DTTFF_COUNT3 );
        cgAddStateEnumerant( state, "Count4",      4); // D3DTTFF_COUNT4 );
        cgAddStateEnumerant( state, "Projected", 256); // D3DTTFF_PROJECTED );

}

// size:
//  -1 = sampler state
//   0 = pass state
//  >0 = srray state

CGstate
RegisterState(const char *name, CGtype type, int size, CGcontext context, int stateId)
{
    CGstate state;

    state = cgGetNamedState( context, name );
    if (state) {
        return state;
    }

    state = cgGetNamedSamplerState( context, name );
    if (state) {
        return state;
    }

    if (size >= 0)
        state = cgCreateArrayState( context, name, type, size );
    else
        state = cgCreateArraySamplerState(context, name, type, 0);

    return state;
}
