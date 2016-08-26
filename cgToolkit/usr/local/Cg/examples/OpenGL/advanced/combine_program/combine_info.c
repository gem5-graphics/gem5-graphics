
/* combine_info.c - deomonstrate how to iterate parameters of a combined
   program using cgGetProgramDomainProgram */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.1 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <string.h>   /* for strcmp */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sqrt, sin, and cos */
#include <assert.h>   /* for assert */
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */

/* An OpenGL 1.2 define */
#define GL_CLAMP_TO_EDGE                    0x812F

/* A few OpenGL 1.3 defines */
#define GL_TEXTURE_CUBE_MAP                 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP         0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X      0x8515

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgComboProgram;  // Just one program handle!

static const char *myProgramName = "combine_program",
                  *myVertexProgramFileName = "C8E6v_torus.cg",
/* page 223 */    *myVertexProgramName = "C8E6v_torus",
                  *myFragmentProgramFileName = "C8E4f_specSurf.cg",
/* page 209 */    *myFragmentProgramName = "C8E4f_specSurf";

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);

  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n",
      myProgramName, situation, string);
    if (error == CG_COMPILER_ERROR) {
      printf("%s\n", cgGetLastListing(myCgContext));
    }
    exit(1);
  }
}

static void reportCombineProgramInfo(CGprogram program);

int main(int argc, char **argv)
{
  CGprogram programList[2];

  myCgVertexProfile = CG_PROFILE_GLSLV;
  myCgFragmentProfile = CG_PROFILE_GLSLF;

  for (argv++; *argv; argv++) {
    if (!strcmp(*argv, "arbvp1")) {
      myCgVertexProfile = CG_PROFILE_ARBVP1;
      printf("%s: arbvp1 profile requested\n", myProgramName);
    } else
    if (!strcmp(*argv, "arbfp1")) {
      myCgFragmentProfile = CG_PROFILE_ARBFP1;
      printf("%s: arbfp1 profiles requested\n", myProgramName);
    } else 
    {
      printf("%s: unknown option: %s\n", myProgramName, *argv);
      exit(1);
    }
  }

  myCgContext = cgCreateContext();
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  checkForCgError("creating context");

  programList[0] =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      myCgVertexProfile,        /* Profile: OpenGL ARB vertex program */
      myVertexProgramName,      /* Entry function name */
      NULL);                    /* No extra compiler options */
  checkForCgError("creating vertex program from file");

  programList[1] =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
      myFragmentProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");

  /* Combine vertex and fragment programs */
  myCgComboProgram = cgCombinePrograms(2, programList);
  checkForCgError("combining programs");
  assert(2 == cgGetNumProgramDomains(myCgComboProgram));

  reportCombineProgramInfo(myCgComboProgram);

  cgDestroyProgram(programList[0]);
  cgDestroyProgram(programList[1]);
  checkForCgError("destroying original programs after combining");

  return 0;
}

static void reportParameters(CGparameter param)
{
  if (param) {
    while (param) {
      const char *name = cgGetParameterName(param);
      CGtype type = cgGetParameterType(param);
      const char *type_name = cgGetTypeString(type);
      const char *semantic = cgGetParameterSemantic(param);

      printf("    %s %s", type_name, name);
      if (semantic && *semantic != '\0') {
        printf(" : %s", semantic);
      }
      printf(";\n");

      param = cgGetNextParameter(param);
    }
  } else {
    printf("    <none>:\n");
  }
}

static const char *domainString(CGdomain domain)
{
  switch (domain) {
  case CG_VERTEX_DOMAIN:
    return "VERTEX";
  case CG_FRAGMENT_DOMAIN:
    return "FRAGMENT";
  case CG_GEOMETRY_DOMAIN:
    return "GEOMETRY";
  case CG_UNKNOWN_DOMAIN:
  default:
    return "UNKNOWN";
  }
}

static void reportCombineProgramInfo(CGprogram program)
{
  int numDomains = cgGetNumProgramDomains(program);
  CGparameter param;
  int i;

  assert(numDomains == 2);

  /* Expect a combined program to have empty parameter lists; the
     parameters are associated with the domain programs. */
  param = cgGetFirstParameter(program, CG_GLOBAL);
  assert(0 == param);
  param = cgGetFirstParameter(program, CG_PROGRAM);
  assert(0 == param);

  for (i=0; i<numDomains; i++) {
    CGprogram subprog = cgGetProgramDomainProgram(program, i);
    const char *entry = cgGetProgramString(subprog, CG_PROGRAM_ENTRY);
    CGprofile profile = cgGetProgramProfile(subprog);
    const char *profile_name = cgGetProfileString(profile);
    CGdomain domain = cgGetProfileDomain(profile);

    printf("%d: %s PROGRAM %s %s\n", i, domainString(domain), profile_name, entry);
    printf("  Global parameters:\n");
    param = cgGetFirstParameter(subprog, CG_GLOBAL);
    reportParameters(param);
    printf("  Local parameters:\n");
    param = cgGetFirstParameter(subprog, CG_PROGRAM);
    reportParameters(param);
  }
}
