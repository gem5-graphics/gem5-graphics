This Cg example demonstrates the use of cgSetCompilerIncludeString for
populating shader source into the Cg virtual file system.  The compiler
first looks in the virtual file system for satisfying #include statements.
A callback function can be used for deferring shader source population
until it is requested by the compiler.

Cg virtual file system support was added in Cg 2.1

See the documentation for:
  cgSetCompilerIncludeString
  cgSetCompilerIncludeFile
  cgSetCompilerIncludeCallback
  cgGetCompilerIncludeCallback
