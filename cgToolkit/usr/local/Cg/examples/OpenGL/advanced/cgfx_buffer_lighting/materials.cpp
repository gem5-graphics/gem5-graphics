
/* materials.cpp - static data for various materials */

#include "materials.h"

const MaterialInfo materialInfo[] = {
  { "emerald",
    { {   0.0215f,   0.1745f,   0.0215f, 1 },     /* ambient */
      {  0.07568f,  0.61424f,  0.07568f, 1 },     /* diffuse */
      {    0.633f, 0.727811f,    0.633f, 1 },     /* specular */
      {     76.8f,         0,         0, 0 } } }, /* shine */
  { "jade",
    { {    0.135f,   0.2225f,   0.1575f, 1 },     /* ambient */
      {     0.54f,     0.89f,     0.63f, 1 },     /* diffuse */
      { 0.316228f, 0.316228f, 0.316228f, 1 },     /* specular */
      {     12.8f,         0,         0, 0 } } }, /* shine */
  { "obsidian",
    { {  0.05375f,     0.05f,  0.06625f, 1 },     /* ambient */
      {  0.18275f,     0.17f,  0.22525f, 1 },     /* diffuse */
      { 0.332741f, 0.328634f, 0.346435f, 1 },     /* specular */
      {     38.4f,         0,         0, 0 } } }, /* shine */
  { "perl",
    { {     0.25f,  0.20725f,  0.20725f, 1 },     /* ambient */
      {         1,    0.829f,    0.829f, 1 },     /* diffuse */
      { 0.296648f, 0.296648f, 0.296648f, 1 },     /* specular */
      {   11.264f,         0,         0, 0 } } }, /* shine */
  { "ruby",
    { {   0.1745f,  0.01175f,  0.01175f, 1 },     /* ambient */
      {  0.61424f,  0.04136f,  0.04136f, 1 },     /* diffuse */
      { 0.727811f, 0.626959f, 0.626959f, 1 },     /* specular */
      {     76.8f,         0,         0, 0 } } }, /* shine */
  { "turquoise",
    { {      0.1f,  0.18725f,   0.1745f, 1 },     /* ambient */
      {    0.396f,  0.74151f,  0.69102f, 1 },     /* diffuse */
      { 0.297254f,  0.30829f, 0.306678f, 1 },     /* specular */
      {     12.8f,         0,         0, 0 } } }, /* shine */
  { "brass",
    { { 0.329412f, 0.223529f, 0.027451f, 1 },     /* ambient */
      { 0.780392f, 0.568627f, 0.113725f, 1 },     /* diffuse */
      { 0.992157f, 0.941176f, 0.807843f, 1 },     /* specular */
      {  27.8974f,         0,         0, 0 } } }, /* shine */
  { "bronze",
    { {   0.2125f,   0.1275f,    0.054f, 1 },     /* ambient */
      {    0.714f,   0.4284f,  0.18144f, 1 },     /* diffuse */
      { 0.393548f, 0.271906f, 0.166721f, 1 },     /* specular */
      {     25.6f,         0,         0, 0 } } }, /* shine */
  { "chrome",
    { {     0.25f,     0.25f,     0.25f, 1 },     /* ambient */
      {      0.4f,      0.4f,      0.4f, 1 },     /* diffuse */
      { 0.774597f, 0.774597f, 0.774597f, 1 },     /* specular */
      {     76.8f,         0,         0, 0 } } }, /* shine */
  { "copper",
    { {  0.19125f,   0.0735f,   0.0225f, 1 },     /* ambient */
      {   0.7038f,  0.27048f,   0.0828f, 1 },     /* diffuse */
      { 0.256777f, 0.137622f, 0.086014f, 1 },     /* specular */
      {     12.8f,         0,         0, 0 } } }, /* shine */
  { "gold",
    { {  0.24725f,   0.1995f,   0.0745f, 1 },     /* ambient */
      {  0.75164f,  0.60648f,  0.22648f, 1 },     /* diffuse */
      { 0.628281f, 0.555802f, 0.366065f, 1 },     /* specular */
      {     51.2f,         0,         0, 0 } } }, /* shine */
  { "silver",
    { {  0.19225f,  0.19225f,  0.19225f, 1 },     /* ambient */
      {  0.50754f,  0.50754f,  0.50754f, 1 },     /* diffuse */
      { 0.508273f, 0.508273f, 0.508273f, 1 },     /* specular */
      {     51.2f,         0,         0, 0 } } }, /* shine */
  { "black plastic",
    { {         0,         0,         0, 1 },     /* ambient */
      {     0.01f,     0.01f,     0.01f, 1 },     /* diffuse */
      {      0.5f,      0.5f,      0.5f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "cyan plastic",
    { {         0,      0.1f,     0.06f, 1 },     /* ambient */
      {         0, 0.509804f, 0.509804f, 1 },     /* diffuse */
      { 0.501961f, 0.501961f, 0.501961f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "green plastic",
    { {         0,         0,         0, 1 },     /* ambient */
      {      0.1f,     0.35f,      0.1f, 1 },     /* diffuse */
      {     0.45f,     0.55f,     0.45f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "red plastic",
    { {         0,         0,         0, 1 },     /* ambient */
      {      0.5f,         0,         0, 1 },     /* diffuse */
      {      0.7f,      0.6f,      0.6f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "white plastic",
    { {         0,         0,         0, 1 },     /* ambient */
      {     0.55f,     0.55f,     0.55f, 1 },     /* diffuse */
      {      0.7f,      0.7f,      0.7f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "yellow plastic",
    { {         0,         0,         0, 1 },     /* ambient */
      {      0.5f,      0.5f,         0, 1 },     /* diffuse */
      {      0.6f,      0.6f,      0.5f, 1 },     /* specular */
      {        32,         0,         0, 0 } } }, /* shine */
  { "black rubber",
    { {     0.02f,     0.02f,     0.02f, 1 },     /* ambient */
      {     0.01f,     0.01f,     0.01f, 1 },     /* diffuse */
      {      0.4f,      0.4f,      0.4f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
  { "cyan rubber",
    { {         0,     0.05f,     0.05f, 1 },     /* ambient */
      {      0.4f,      0.5f,      0.5f, 1 },     /* diffuse */
      {     0.04f,      0.7f,      0.7f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
  { "green rubber",
    { {         0,     0.05f,         0, 1 },     /* ambient */
      {      0.4f,      0.5f,      0.4f, 1 },     /* diffuse */
      {     0.04f,      0.7f,     0.04f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
  { "red rubber",
    { {     0.05f,         0,         0, 1 },     /* ambient */
      {      0.5f,      0.4f,      0.4f, 1 },     /* diffuse */
      {      0.7f,     0.04f,     0.04f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
  { "white rubber",
    { {     0.05f,     0.05f,     0.05f, 1 },     /* ambient */
      {      0.5f,      0.5f,      0.5f, 1 },     /* diffuse */
      {      0.7f,      0.7f,      0.7f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
  { "yellow rubber",
    { {     0.05f,     0.05f,         0, 1 },     /* ambient */
      {      0.5f,      0.5f,      0.4f, 1 },     /* diffuse */
      {      0.7f,      0.7f,     0.04f, 1 },     /* specular */
      {        10,         0,         0, 0 } } }, /* shine */
};
const int materialInfoCount = sizeof(materialInfo)/sizeof(materialInfo[0]);
