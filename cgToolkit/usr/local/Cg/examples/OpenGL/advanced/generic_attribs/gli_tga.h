#ifndef __gli_tga_h__
#define __gli_tga_h__

/* gli_tga.h - interface for TrueVision (TGA) image file loader */

/* Copyright NVIDIA Corporation, 1999. */

/* A lightweight TGA image file loader for OpenGL programs. */

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef struct {
  uint8 idLength;
  uint8 colorMapType;

  /* The image type. */
#define TGA_TYPE_MAPPED 1
#define TGA_TYPE_COLOR 2
#define TGA_TYPE_GRAY 3
#define TGA_TYPE_MAPPED_RLE 9
#define TGA_TYPE_COLOR_RLE 10
#define TGA_TYPE_GRAY_RLE 11
  uint8 imageType;

  /* Color Map Specification. */
  /* We need to separately specify high and low bytes to avoid endianness
     and alignment problems. */
  uint8 colorMapIndexLo, colorMapIndexHi;
  uint8 colorMapLengthLo, colorMapLengthHi;
  uint8 colorMapSize;

  /* Image Specification. */
  uint8 xOriginLo, xOriginHi;
  uint8 yOriginLo, yOriginHi;

  uint8 widthLo, widthHi;
  uint8 heightLo, heightHi;

  uint8 bpp;

  /* Image descriptor.
     3-0: attribute bpp
     4:   left-to-right ordering
     5:   top-to-bottom ordering
     7-6: zero
     */
#define TGA_DESC_ABITS 0x0f
#define TGA_DESC_HORIZONTAL 0x10
#define TGA_DESC_VERTICAL 0x20
  uint8 descriptor;

} TgaHeader;

typedef struct {
  uint32 extensionAreaOffset;
  uint32 developerDirectoryOffset;
#define TGA_SIGNATURE "TRUEVISION-XFILE"
  uint8 signature[16];
  uint8 dot;
  uint8 null;
} TgaFooter;

#endif /* __gli_tga_h__ */
