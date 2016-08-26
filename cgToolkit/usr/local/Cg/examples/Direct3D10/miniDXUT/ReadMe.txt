ReadMe.txt
April 07, 2008
--------------

This version of DXUT called miniDXUT is based on the December 2005
DirectX SDK version of DXUT.

This DXUT code is modified so:

*  Only the initialization, window creation, and event handling
   functionality is provided.

*  miniDXUT no longer depends on the D3DX9 DLL.

Look for the MINI_DXUT ifdef's in the code to see the modifications.

Only the DXUT.cpp, DXUTenum.cpp, and DXUTmisc.cpp source files are
necessary for miniDXUT.  Only the DXUT.h, DXUTenum.h, and DXUTmisc.h
headers are necessary for miniDXUT.
